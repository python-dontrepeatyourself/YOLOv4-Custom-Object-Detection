import cv2
import glob


# define the minimum confidence (to filter weak detections), 
# Non-Maximum Suppression (NMS) threshold, the green color, and the label
confidence_thresh = 0.5
NMS_thresh = 0.3
green = (0, 255, 0)
label = "Raccoon"

# get the list of all the images in the test folder
images_list = glob.glob("Raccoon/test/*.jpg")

for image_path in images_list:
    image = cv2.imread(image_path)

    # get the image dimensions
    h = image.shape[0]
    w = image.shape[1]
        
    # load the configuration and weights files from disk
    yolo_config = "yolov4-tiny-custom-raccoon.cfg"
    yolo_weights = "yolov4-tiny-custom-raccoon_final.weights"

    # load the YOLOv4 network pre-trained on our raccoon dataset
    net = cv2.dnn.readNetFromDarknet(yolo_config, yolo_weights)

    # Get the name of all the layers in the network
    layer_names = net.getLayerNames()
    # Get the names of the output layers
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # create a blob from the image
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255, (416, 416), swapRB=True, crop=False)
    # pass the blob through the network and get the output predictions
    net.setInput(blob)
    outputs = net.forward(output_layers)

    # create empty lists for storing the bounding boxes and confidences
    boxes = []
    confidences = []

    # loop over the output predictions
    for output in outputs:
        # loop over the detections
        for detection in output:
            # get the confidence of the dected object
            confidence = detection[5]

            # we keep the bounding boxes if the confidence (i.e. class probability) 
            # is greater than the minimum confidence 
            if confidence > confidence_thresh:
                # perform element-wise multiplication to get
                # the coordinates of the bounding box
                box = [int(a * b) for a, b in zip(detection[0:4], [w, h, w, h])]
                center_x, center_y, width, height = box
                
                # get the top-left corner of the bounding box
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))

                # append the bounding box and the confidence to their respective lists
                confidences.append(float(confidence))
                boxes.append([x, y, width, height])

    # apply non-maximum suppression to remove weak bounding boxes that overlap with others.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_thresh, NMS_thresh)
    
    # loop over the indices only if the `indices` list is not empty
    if len(indices) > 0:
        # loop over the indices
        for i in indices.flatten():
            (x, y, w, h) = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
            text = f"{label}: {confidences[i] * 100:.2f}%"
            cv2.putText(image, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green, 2)

    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)