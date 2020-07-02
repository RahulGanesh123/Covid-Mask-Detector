# import the necessary packages
import time
import cv2
import numpy as np
import os


def get_yolo_net(cfg_path, weight_path):
    """
    return YOLO net.
    run this function when app starts to load the net.
    """

    if not cfg_path or not weight_path:
        raise Exception('missing inputs. See file.')

    print('[INFO] loading YOLO from disk...')
    net = cv2.dnn.readNetFromDarknet(cfg_path, weight_path)

    return net

def yolo_forward(net, LABELS, image, confidence_level, save_image=False):
    """
    forward data through YOLO network
    """

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(10000, 3),
                               dtype='uint8')

    # grab image spatial dimensions
    (H, W) = image.shape[:2]

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    # also time it
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layer_outputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    print('[INFO] YOLO took {:.6f} seconds'.format(end - start))

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    class_ids = []

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_level:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')

                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    # idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_level, threshold)

    print(class_ids)
    print(LABELS)
    # print(labels)

    labels = [LABELS[i] for i in class_ids]

    if save_image:
        yolo_save_img(image, class_ids, boxes, labels, confidences, colors, 'python_predictions.jpg')

    return class_ids, labels, boxes, confidences


def yolo_save_img(image, class_ids, boxes, labels, confidences, colors, file_path):
    """
    save a image with bounding boxes
    """
    for i, box in enumerate(boxes):
        # extract the bounding box coordinates
        (x, y) = (box[0], box[1])
        (w, h) = (box[2], box[3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        text = '{}'.format(labels[i])
        # text = '{}: {:.4f}'.format(labels[i], confidences[i])
        print(text)

        font_scale = 1.3
        # set the rectangle background to white
        rectangle_bgr = color
        # set some text
        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = x
        text_offset_y = y - 3 
        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10    ))
        cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=font_scale, color=(255, 255, 255), thickness=2)
    cv2.imwrite(file_path, image)
    return image


def yolo_show_img(image, class_ids, boxes, labels, confidences, colors):
    """
    show without save a image with bounding boxes
    """
    for i, box in enumerate(boxes):
        # extract the bounding box coordinates
        (x, y) = (box[0], box[1])
        (w, h) = (box[2], box[3])

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 3)
        text = '{}: {:.4f}'.format(labels[i], confidences[i])
        print(text)

        font_scale = 1.3
        # set the rectangle background to white
        rectangle_bgr = color
        # set some text
        # get the width and height of the text box
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=1)[0]
        # set the text start position
        text_offset_x = x
        text_offset_y = y - 3 
        # make the coords of the box with a small padding of two pixels
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 10, text_offset_y - text_height - 10    ))
        cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
        cv2.putText(image, text, (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=font_scale, color=(255, 255, 255), thickness=2)

    cv2.imshow('yolo prediction', image)
    cv2.waitKey(0)


def yolo_pred(image_path, names_path, cfg_path, weight_path):
    # get the net using cfg and weight
    net = get_yolo_net(cfg_path, weight_path)

    # prepare labels and colors
    labels = open(names_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # read images
    image = cv2.imread(image_path)

    (class_ids, labels, boxes, confidences) = yolo_forward(
        net, labels, image, confidence_level=0.5)

    yolo_show_img(image, class_ids, boxes, labels, confidences, colors)



def yolo_pred_list(image_folder_path, names_file, cfg_file, weight_file, confidence_level=0.5, threshold=0.3, save_image=False):

    all_paths = os.listdir(image_folder_path)
    image_paths = [os.path.join(image_folder_path, f) \
        for f in all_paths if '.jpg' in f or '.png' in f]

    image_paths.sort()

    LABELS = open(names_file).read().strip().split("\n")

    # loading yolo net
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(cfg_file, weight_file)

    # make predictions
    output = []    
    for image_path in image_paths:
        print('++++++++++New Prediction+++++++++')
        print(image_path)
        image = cv2.imread(image_path)
        (class_ids, labels, boxes, confidences) = yolo_forward(
            net, LABELS, image, confidence_level, save_image=save_image)
        result = {
            'image_path': image_path,
            'class_ids': class_ids,
            'labels': labels,
            'boxes': boxes,
            'confidences': confidences
        }
        output.append(result)
    
    return output

def yolo_video(name_path, cfg_path, weight_path):
    # get the net using cfg and weight
    net = get_yolo_net(cfg_path, weight_path)

    # prepare labels and colors
    LABELS = open(name_path).read().strip().split('\n')
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    cv2.namedWindow('yolo prediction')
    while True:
        ret, image = cam.read()
        # time.sleep(1)
        (class_ids, labels, boxes, confidences) = yolo_forward(
            net, LABELS, image, confidence_level=0.3)

        if len(class_ids) > 0:
            for i, box in enumerate(boxes):
                # extract the bounding box coordinates
                (x, y) = (box[0], box[1])
                (w, h) = (box[2], box[3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = '{}: {:.4f}'.format(labels[i], confidences[i])
                print(text)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('yolo prediction', image)
        print('video mode')
        k = cv2.waitKey(1)

        if k == 27:  # Esc key to stop
            cv2.waitKey(1)
            cam.release()
            break

    cv2.destroyAllWindows()
    cv2.waitKey(100)
