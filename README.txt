previous_version/editData.py: data processing tools, such as modcrop() and inputGen().

previous_version/wyconMain.py: initiate the parameters and the wyconSRCNN model for training and testing.

previous_version/wyconSRCNN.py: 3-layer CNN model implementing SRCNN system.

-------------------------------------------------------------------------------------------------------------------
For training:
enter in command line: python wyconMain.py

For testing:
enter in conmmand line: python wyconMain.py --is_train=False --stride=21

-------------------------------------------------------------------------------------------------------------------

wySR.py: New version of the implementation of SRCNN.

wyTool.py: New version of data processing tools.

wyMain.py: Initiate the wySR instance to reconstruct images.

psnr.py: Comput the psnr value of a prediction.

-------------------------

For training:

enter in command line: python wyMain.py

For testing:

enter in conmmand line: python wyMain.py --is_train=False --stride=21

For reconstructing cifar images:

enter in conmmand line: python wyMain.py --is_train=False --stride=21 --cifar=True

The reconstruction target could be changed by modifying the line 148 in wyTool.py
For example:
	change to: input, label = preprocess(data[2], scale, num_chnl, cifar)





