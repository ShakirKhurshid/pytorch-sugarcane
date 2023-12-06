import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision
from torch.utils.data.sampler import SubsetRandomSampler
import argparse


def process_arguments():
	'''Collect the input argument's according to the syntax
		Return a parser with the arguments
	'''

	parser = argparse.ArgumentParser(description = 'Train the model on a the dataset and save the model')


    parser.add_argument('--data_directory',
		                action='store',
		                type = str,
		                required = True,
		                default='data/train',
		                help='Input directory for training data')

    parser.add_argument('--save_dir',
		                action='store',
		                type = str,
		                dest='save_directory',
		                default='checkpoint_dir',
		                help='Directory where the checkpoint file is saved')

    parser.add_argument("--output_dir",
    					default=None, 
    					type=str, 
    					required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )

    parser.add_argument('--epochs',
                   		action='store',
                   		dest='epochs', 
                   		type=int, 
                   		default=5,
                   		help='Number of Epochs for the training')

    parser.add_argument('-wr', 
    					'--warm-restart', 
    					dest='warm_restart', 
                    	action='store_true',
                    	help='Passing this argument while training will ensure that Cosine Annealing with Warm Restarts is used while training')

	parser.add_argument('-t0', 
						'--t-zero', 
						dest='t_zero', 
						type=int,
                    	default=5,
                    	help='The initial number of epochs for the first warm restart.'
                    	)


	parser.add_argument('-tm', 
						'--t-mult', 
						dest='t_mult', 
						type=int,
                    	default=1,
                    	help='The multiplicative factor for the number of epochs for the warm restart')


# Get the input parameters and train the specific network
def main():
	# Device configuration
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# Specify transforms using torchvision.transforms as transforms library
	transformations = transforms.Compose([
	    transforms.Resize(255),
	    transforms.CenterCrop(224),
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	    ])


	#load Training and validation data
	train_set = datasets.ImageFolder(input_arguments.data_directory+ '/train', transform = transformations)
	val_set = datasets.ImageFolder(input_arguments.data_directory+ '/val', transform = transformations)
	class_names  = train_set.classes


	# Creating Data_Loader with a batch size of 32
	train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
	val_loader = torch.utils.data.DataLoader(val_set, batch_size =32, shuffle=True)


	# Load a Pretrained Model Resnet-50
	model = models.resnet50(pretrained= True) # Set False if you want to train the completetly on your own dataset


	#Using the pretrained model we Dont Train the initial layers.
	for param in model.parameters():
	    param.requires_grad = True #Set True to train the whole network


	# Creating final fully connected Layer that accorting to the no of classes we require
	model.fc = nn.Sequential(nn.Linear(2048, 512),
	                        nn.ReLU(),
	                        nn.Dropout(0.2),
	                        nn.Linear(512,len(class_names)),
	                        nn.LogSoftmax(dim=1))
	model.to(device)


	# Loss and optimizer
	criterion = nn.NLLLoss()

	# In order to apply layer-wise learning , diffrential learnning
	optimizer = optim.SGD([
					      {'params': model.conv1.parameters(), 'lr':1e-4},
					      {'params': model.layer1.parameters(), 'lr':1e-4},
					      {'params': model.layer2.parameters(),'lr':1e-4},
					      {'params': model.layer3.parameters(),'lr':1e-3},
					      {'params': model.layer4.parameters() ,'lr':1e-3},
					      {'params': model.fc.parameters(), 'lr': 1e-2}   # the classifier needs to learn weights faster
	                      ], 
	                      lr=0.001, weight_decay=0.0005
	                      )


	# Restarts the learning rate after every 5 epoch
	scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
															        optimizer, 
															        T_0= 5, # Number of iterations for the first restart.
															        T_mult= 1, #  A factor increases Ti after the restart
	    															)


	epochs = input_arguments.epochs
	best_acc = 0.0
	iters = len(train_loader)
	train_loss, val_loss = [], []
	for epoch in range(epochs):
	    train_loss_epoch = 0
	    valid_loss_epoch = 0
	    accuracy = 0


	    # Traininng the model
	    model.train()
	    counter = 0
	    for i, sample in enumerate(train_loader):
	        inputs, labels = sample

	        # Move to device
	        inputs, labels = inputs.to(device), labels.to(device)

	        # Clear Optimizers
	        optimizer.zero_grad()

	        # Forward Pass
	        logps = model.forward(inputs)

	        # Loss
	        loss = criterion (logps, labels)

	        # Backprop (Calculate Gradients)
	        loss.backward()

	        # Adjust parameters based on gradients
	        optimizer.step()

	        # Reduce the LR with Cosine Annealing
	        scheduler.step(epoch + i/iters)


	        # Add the loss to the trainininputs.size ( 0 ) set's running loss
	        train_loss_epoch += loss.item() * inputs.size(0)


	        # Print the progress of our training
	        counter += 1
	        print(counter, "/", len(train_loader))


	    # Evaluating the model     
	    model.eval ()
	    counter = 0

	    # Tell torch not to calculate gradients
	    with torch.no_grad ():
	        for inputs, labels in val_loader:
	            
	            # Move to device
	            inputs, labels = inputs.to(device), labels.to(device)
	            
	            # Forward pass
	            output = model.forward(inputs)
	            
	            # Calculate Loss
	            valloss = criterion(output, labels)
	            
	            # Add loss to the validation set's running loss
	            valid_loss_epoch += valloss.item() * inputs.size(0)
	            
	            # Since our model outputs a LogSoftmax, find the real
	            # percentages by reversing the log function
	            output = torch.exp(output)
	            
	            # Get the top class of the output
	            top_p, top_class = output.topk (1, dim=1)

	            # See how many of the classes were correct?
	            equals = top_class == labels.view (*top_class.shape)

	            # Calculate the mean (get the accuracy for this batch)
	            # and add it to the running accuracy for this epoch
	            accuracy += torch.mean (equals.type (torch.FloatTensor)).item ()

	            # Print the progress of our evaluation
	            counter += 1
	            print (counter, "/", len(val_loader))

	        # Save_the_best _accuracy_model
	        if (accuracy / len (val_loader)) > best_acc:
	            best_acc = accuracy / len (val_loader)

	            #Create a file path using the specified save_directory
	            #to save the file as checkpoint.pth under that directory
	            if not os.path.exists(input_arguments.save_directory):
        			os.makedirs(input_arguments.save_directory)
    				checkpoint_file_path = os.path.join(input_arguments.save_directory, 'RESNET-50'+"_"+str(input_arguments.epochs)+".pth")
	            torch.save ( model.state_dict(), checkpoint_file_path)

	    # Get the average loss for the entire epoch
	    train_loss_epoch = train_loss / len(train_loader.dataset)
	    valid_loss_epoch = val_loss / len(val_loader.dataset)

	    # Print out the information
	    print ('\nEpoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format (epoch, train_loss_epoch, valid_loss_epoch))
	    print ('Accuracy: ', accuracy / len(val_loader))

	    train_loss.append(train_loss_epoch)
	    val_loss.append(valid_loss_epoch)

	print ('\nBest Accuracy', best_acc)



##########--------------Evaluation Metrcs on Test set -------------------######


#Load test data :

'''
# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1


# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                   accuracy))


