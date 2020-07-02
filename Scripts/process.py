# Libraries
import json
import os
import pandas as pd
import requests 
import traceback
from PIL import Image

#download images from labelbox
def dl_img(path_to_train, prefix_filenm, image_url, idx):
    print("Downloading: {}".format(image_url))
    response_img = requests.get(image_url)
    if not response_img:
        raise Exception(" there is no response from url.")
    img_filenm = prefix_filenm + str(idx) + ".jpg"
    '''
    img_filenm = ""
    if response_img.content.format =='JPEG':
        img_filenm = "rockfish_" + str(idx) + ".jpg"
    elif response_img.format == 'PNG':
        img_filenm = "rockfish_" + str(idx) + ".png"
    elif response_img.format == 'GIF':
        img_filenm = "rockfish_" + str(idx) + ".gif"
    else:
        img_filenm = "unknown_rockfish_" + str(idx) + ".dud"
        print("Image is of unidentified type.")
    '''
    with open(os.path.join(path_to_train, img_filenm),'wb') as img_file:
        img_file.write(response_img.content)
        print("Saved image at this location: {}".format(os.path.join(path_to_train, img_filenm)))
    return img_filenm

# get image from local and find out the dimensions
def get_img_dim(path_to_train, img_filenm):
    image_dimensions = 0, 0
    with Image.open(os.path.join(path_to_train, img_filenm)) as image: 
        image_dimensions = image.size 
    return image_dimensions

#get coordinates of boxes
def get_coordinates(a_box, key):
    list_coords = []
    for vertices in a_box[key]:
        coords = []
        coords.append(vertices.get('x'))
        coords.append(vertices.get('y'))
        list_coords.append(coords)
    return list_coords


#returns all the boxes with their coordinates
def get_boxes(lbl_set, key):
    obj_count = 1
    box_key ='geometry'
    list_boxes = []
    for info in lbl_set[key]:
        list_boxes.append(get_coordinates(info, box_key))
        obj_count += 1
    #print(" These are the list of boxes with coordinates : \n {}".format(list_boxes))
    return list_boxes


# method for creating a new train.txt file
def create_file_txt(dest_dir,file_name):
    file_path_nm = os.path.join(dest_dir, file_name)
    try:
        with open(file_path_nm, "w") as train_txt:
            print("Created {}".format(file_path_nm))
    except Exception as e:
        raise e("Error while creating {}".format(file_path_nm))


# method to adding image path and name to train.txt
def add_img_to_train_txt(dest_dir, train_file, relative_lblpath, file_name):
    train_txt_path = os.path.join(dest_dir, train_file)
    try:
        with open(train_txt_path, "a") as train_txt:
            train_txt.write("{}\n".format(os.path.join(relative_lblpath, file_name)))
            print("\tAdded {}".format(file_name) + " to {}".format(train_txt_path))
    except Exception as e:
        raise e("Could not add img path to {}".format(train_txt_path))

#yolo format from box coord
def get_yolo_format(obj_type, box, img_dims):
    list_x = []
    list_y = []
    for coord in box:
        list_x.append(coord[0])
        list_y.append(coord[1])
    dw = 1./img_dims[0]
    dh = 1./img_dims[1]
    center_x = dw * (max(list_x) + min(list_x))/2.0
    center_y = dh * (max(list_y) + min(list_y))/2.0
    rel_width = dw * (max(list_x) - min(list_x))  
    rel_height = dh * (max(list_y) - min(list_y)) 
    line_to_write = f"{obj_type} {center_x} {center_y} {rel_width} {rel_height}\n"
    return line_to_write


#saves the yolo stuff in the text file(remember 1.jpg --> 1.txt)
def save_box_info(obj_type, labels, text, img_dims):
    count = 1
    for box in labels:
        # print("We are now going through our label # {} with content {}".format(count,box))
        box_line = get_yolo_format(obj_type, box, img_dims)
        #print("Writing line {}".format(box_line))
        text.write(box_line)
        count += 1
    print("Total of {} boxes written to txt file".format(count))


def main():
    lbl_bxfile = "lblbx_data.csv"
    train_file = "train.txt"
    path_to_train = "./data_2_train/"
    path_to_test = "./data_4_test/"
    prefix_path ="./"
    prefix_filenm = "rockfish_"
    relative_lblpath = "C:\Users\rahul\Projects\Covid-Mask-Detector\Scripts\img"
    label_type_0 = 'Face_with_masks'
    label_type_1 = 'Face_no_masks'
    create_file_txt(prefix_path, train_file)
    df = pd.read_csv(lbl_bxfile)
    total_rows = len(df.index) 
    print("There are a total of {} images to process.".format(total_rows))
    total_rows = 3
    for idx in range(0, total_rows):
        json_acceptable_string = df["Label"][idx].replace("'", "\"")
        if( json_acceptable_string == 'Skip'):
            print("This is the label here is {} . We will skip this label.".format(json_acceptable_string))
            continue
        label_info = json.loads(json_acceptable_string)
        if not label_info:
            print(" At Index {}, there is no label info".format(idx))
            continue
        # download image and get width and height
        image_url = df["Labeled Data"][idx]
        img_filenm = dl_img(path_to_train, prefix_filenm, image_url, idx)
        img_dims = get_img_dim(path_to_train, img_filenm)    
        add_img_to_train_txt(prefix_path, train_file, relative_lblpath, img_filenm)
        # create label.txt file
        img_txtnm = prefix_filenm + str(idx) + ".txt"
        text = open(os.path.join(path_to_train, img_txtnm), "w+")
        if label_type_0 in label_info:
            print("test")
            labels = get_boxes(label_info, label_type_0)
            save_box_info(0, labels , text, img_dims)   
        if  label_type_1 in label_info:
            labels = get_boxes(label_info, label_type_1)
            save_box_info(1, labels, text, img_dims)  
        text.close()
        print("finished index #: {}".format(idx))
if __name__ == '__main__':
    try:
        main()
        print('========================End of program===============================')
    except Exception as e:
        raise e("Unknown error in obtaining Labelbox project data set ")
        traceback.print_exc()