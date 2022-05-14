import os
import cv2
import numpy as np
import PySimpleGUI as sg

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def zoom(img):
    value = np.random.uniform(0.7, 1)
    h, w = img.shape[:2]
    h_new = int(value*h)
    w_new = int(value*w)
    h_start = np.random.randint(0, h-h_new)
    w_start = np.random.randint(0, w-w_new)
    img = img[h_start : h_start + h_new, w_start :w_start + w_new, : ]
    return img

def rotation(img):
    angle = np.random.uniform(-90, 90)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    img = cv2.warpAffine(img, M, (w, h), borderValue = (255, 255, 255))
    return img

def brightness(img):
    value = np.random.uniform(0.3, 2)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype = np.float64)
    hsv[:, :, 1] = hsv[:, :, 1] * value
    hsv[:, :, 1][hsv[:, :, 1] > 255]  = 255
    hsv[:, :, 2] = hsv[:, :, 2] * value 
    hsv[:, :, 2][hsv[:, :, 2] > 255]  = 255
    hsv = np.array(hsv, dtype = np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

def horizontal_shift(img):
    ratio = np.random.uniform(-0.25, 0.25)
    w = img.shape[1]
    shift = int(w * ratio)
    if ratio > 0:
        img = img[:, :w - shift, :]
        img = cv2.copyMakeBorder(img, 0, 0, shift, 0, cv2.BORDER_CONSTANT,value = (255, 255, 255))
    if ratio < 0:
        img = img[:, -1 * shift:, :]
        img = cv2.copyMakeBorder(img, 0, 0, 0, -1 * shift, cv2.BORDER_CONSTANT,value = (255, 255, 255))
    return img

def vertical_shift(img):
    ratio = np.random.uniform(-0.25, 0.25)
    h = img.shape[0]
    shift = int(h * ratio)
    if ratio > 0:
        img = img[:h - shift, :, :]
        img = cv2.copyMakeBorder(img, shift, 0, 0, 0, cv2.BORDER_CONSTANT, value = (255, 255, 255))
    if ratio < 0:
        img = img[-1 * shift:, :, :]
        img = cv2.copyMakeBorder(img, 0, -1 * shift, 0, 0, cv2.BORDER_CONSTANT, value = (255, 255, 255))
    return img

def horizontal_flip(img):
    return cv2.flip(img, 1)
    
def vertical_flip(img):
    return cv2.flip(img, 0)

def resize_image(original_image, size_of_image):
    width, height = size_of_image
    scale = max(original_image.shape[1]/width, original_image.shape[0]/height)
    w, h = int(original_image.shape[1]/scale), int(original_image.shape[0]/scale)
    original_image = cv2.resize(original_image, (w, h))
    delta_w = abs(width - w)
    delta_h = abs(height - h)
    original_image = cv2.copyMakeBorder(original_image, delta_h//2, delta_h - delta_h//2, delta_w//2, delta_w - delta_w//2, cv2.BORDER_CONSTANT, value = (255, 255, 255))
    return original_image

def save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path):
    if i <= images_to_generate:
        result_image = resize_image(original_image, size_of_image)
        new_image_path = "%s/augmented_image_%s.jpg" %(augmented_path, i)
        cv2.imwrite(new_image_path, result_image)
        if mergeflag == False:
            original_image = cv2.imread(image)
            i = i + 1
            if firstflag:
                im_list.append(result_image)
    return result_image, original_image, i, im_list

def make_collage(imgs, default_size_of_image):
    i, n = 0, 3
    padimg = np.zeros((default_size_of_image[0], default_size_of_image[1], 3), dtype = np.uint8)
    padmat = [[padimg for i in range(n)] for j in range(n)]
    for j in range(n):
        for k in range(n):
            if i < len(imgs):
                imgs[i] = resize_image(imgs[i], default_size_of_image)
                padmat[j][k] = imgs[i]
                i = i + 1
            else:
                break
    imgs_2d = cv2.vconcat((cv2.hconcat(padmat[0]), cv2.hconcat(padmat[1]), cv2.hconcat(padmat[2])))
    imgs_2d = cv2.resize(imgs_2d, default_size_of_image)
    return imgs_2d

def user_interface():
    cancelflag = 0
    foldername = None
    augmented_path = None
    sg.theme('DarkBlack')
    default_size_of_image = (400, 400)

    left_col = [[sg.Image(background_color = 'black', size = default_size_of_image, key = '-IMAGE-', expand_x = True, expand_y = True)],
                [sg.Text('', font = ("Defualt", 16), key = 'text1')]]

    right_col = [[sg.Image(background_color = 'black', size = default_size_of_image, key = '-IMAGE1-', expand_x = True, expand_y = True)],
                [sg.Text('', font = ("Defualt", 16), key = 'text2')]]

    layout = [[sg.Button('Select Folder', tooltip = 'Click here to select your dataset folder.'), 
               sg.Input(disabled = True, text_color = 'black', expand_x = True, background_color = 'white', tooltip = 'Dataset folder', key = 'foldername'), 
               sg.VerticalSeparator(),
               sg.Text('Enter Number of Output Images'), 
               sg.InputText(text_color = 'black', background_color = 'white', size =(15, 1), tooltip = 'Number of images to generate.', key = 'Number of Output Images'), 
               sg.Button('Augment', tooltip = 'Click here to start the augmentation process.')],
              [sg.HorizontalSeparator()],
              [sg.Checkbox('Merge Augmentations', enable_events = True, tooltip = 'Check this box to merge all the selected augmentation types.', key='Merge'), 
               sg.Checkbox('Check all', enable_events = True, tooltip = 'Check this box to check all augmentation types.', key = 'Check_All'), 
               sg.Checkbox('Uncheck all', enable_events = True, tooltip = 'Check this box to uncheck all augmentation types.', key='Uncheck_All'),
               sg.VerticalSeparator(),
               sg.Text('Width'), sg.InputText(text_color = 'black', background_color = 'white', size = (10, 1), tooltip = 'Output image width.', key = 'Width'),
               sg.Text('Height'), sg.InputText(text_color = 'black', background_color = 'white', size = (10, 1), tooltip = 'Output image height.', key = 'Height'),
               sg.VerticalSeparator(),
               sg.Button('Show Folder', disabled = True, tooltip = 'Open output folder.', key = 'Show Folder')],
              [sg.HorizontalSeparator()],
              [sg.Checkbox('Original', enable_events = True, tooltip = 'Check this box to return the image unchanged.', key = 'check Original'), 
               sg.Checkbox('Gray', enable_events = True, tooltip = 'Check this box to return the image in grayscale.', key = 'check Gray'),
               sg.Checkbox('Zoom', enable_events = True, tooltip = 'Check this box to return the image zoomed.', key = 'check Zoom'), 
               sg.Checkbox('Rotation', enable_events = True, tooltip = 'Check this box to return the image rotated.', key = 'check Rotation'), 
               sg.Checkbox('Brightness', enable_events = True, tooltip = 'Check this box to return the image with different brightness.', key = 'check Brightness'), 
               sg.Checkbox('Horizontal Shift', enable_events = True, tooltip = 'Check this box to return the image shifted horizontally.', key = 'check Horizontal Shift'), 
               sg.Checkbox('Vertical Shift', enable_events = True, tooltip = 'Check this box to return the image shifted vertically.', key = 'check Vertical Shift'), 
               sg.Checkbox('Horizontal Flip', enable_events = True, tooltip = 'Check this box to return the image flipped horizontally.', key = 'check Horizontal Flip'),
               sg.Checkbox('Vertical Flip', enable_events = True, tooltip = 'Check this box to return the image flipped vertically.', key = 'check Vertical Flip')],
              [sg.HorizontalSeparator()],
              [sg.Column(left_col, vertical_alignment = 'center', justification = 'center',  k = '-C-', element_justification = 'c'), 
               sg.VSeperator(),sg.Column(right_col, vertical_alignment = 'center', justification = 'center',  k = '-C-', element_justification = 'c')]]

    window = sg.Window("Augmento", layout, size = (900, 600), icon = 'icon.ico', enable_close_attempted_event=True, finalize=True, element_justification='c')

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, "Exit"):
            break
        
        elif event == 'Select Folder':
            foldername = sg.popup_get_folder("Please select your dataset folder.", keep_on_top = True, no_titlebar = True, no_window = True)
            window['foldername'].update(foldername)
            if foldername and os.path.isdir(foldername):
                original_path = foldername
                augmented_path = foldername + "/results"
                window['Show Folder'].update(disabled = True)
                images = []
                for im in os.listdir(original_path):
                    if im.lower().endswith(('.png', '.jpg', '.jpeg')):
                        images.append(os.path.join(original_path, im))

        elif event == 'Augment':
            if foldername and os.path.isdir(foldername) and len(images) > 0:
                if ((values['check Original'] or values['check Zoom'] or values['check Rotation'] or values['check Brightness'] or values['check Gray'] or 
                     values['check Horizontal Shift'] or values['check Vertical Shift'] or values['check Horizontal Flip'] or values['check Vertical Flip']) == False):
                    sg.Popup('Please check at least one checkbox!.', keep_on_top = True, no_titlebar = True)
                    continue
                
                if augmented_path and os.path.isdir(augmented_path):
                    for file in os.listdir(augmented_path):
                        os.remove(os.path.join(augmented_path, file)) 
                    os.rmdir(augmented_path)
                                
                textbox = values['Number of Output Images']    
                if textbox == '':
                    sg.Popup('Null Input!\nDefault number of (200) output images will be used.', keep_on_top = True, no_titlebar = True)
                    images_to_generate = 200
                    window['Number of Output Images'].update(images_to_generate)
                
                else:
                    try:
                        images_to_generate = int(textbox)
                        window['Number of Output Images'].update(images_to_generate)
                        
                    except:
                        sg.Popup("Invalid Input, Integers ONLY!\nDefault number of (200) output images will be used.", keep_on_top = True, no_titlebar = True)
                        images_to_generate = 200
                        window['Number of Output Images'].update(images_to_generate)
                        
                w, h = size_of_image = (values['Width'], values['Height'])    
                if w == '' or h == '':
                    sg.Popup('Null Input!\nDefault resolution of (400 x 400) will be used.', keep_on_top = True, no_titlebar = True)
                    w, h = size_of_image = default_size_of_image
                    window['Width'].update(w)
                    window['Height'].update(h)
                    
                else:
                    try:
                        w, h = size_of_image = (int(w), int(h))
                        window['Width'].update(w)
                        window['Height'].update(h)
                        
                    except:
                        sg.Popup("Invalid Input, Integers ONLY!\nDefault resolution of (400 x 400) is used.", keep_on_top = True, no_titlebar = True)
                        w, h = size_of_image = default_size_of_image
                        window['Width'].update(w)
                        window['Height'].update(h)
                        
                start = sg.popup_yes_no('The augmentation process is about to begin!\n(' + str(len(images)) + ') images as input.\n(' + str(images_to_generate) + 
                                        ') images as output.', keep_on_top = True, no_titlebar = True)
                    
                if start == 'No':
                    cancelflag = 1
                    continue
                
                os.mkdir(augmented_path)
                window['Show Folder'].update(disabled = False)
                i = 1
                mergeflag = values["Merge"]
                im_list = []
                
                while i <= images_to_generate:    
                    image = np.random.choice(images)
                    original_image = result_image = cv2.imread(image)
                    cancelflag = 0
                    firstflag = 0
                    
                    if i == 1:
                        result_image = resize_image(original_image, default_size_of_image)
                        result_image = cv2.imencode('.png', result_image)[1].tobytes()
                        window['-IMAGE-'].update(data = result_image)
                        window['text1'].update('Original Sample Image')
                        firstflag = 1
                                        
                    if values["check Original"] == True:
                        original_image = cv2.imread(image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                    
                    if values["check Gray"] == True:
                        original_image = grayscale(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                
                    if values["check Zoom"] == True:
                        original_image = zoom(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                    
                    if values["check Rotation"] == True:
                        original_image = rotation(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                                                    
                    if values["check Brightness"] == True:
                        original_image = brightness(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)       

                    if values["check Horizontal Shift"] == True:
                        original_image = horizontal_shift(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)  
                        
                    if values["check Vertical Shift"] == True:
                        original_image = vertical_shift(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                                        
                    if values["check Horizontal Flip"] == True:
                        original_image = horizontal_flip(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                            
                    if values["check Vertical Flip"] == True:
                        original_image = vertical_flip(original_image)
                        result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path)
                                                           
                    if mergeflag == True:
                        if firstflag:
                            result_image = resize_image(result_image, default_size_of_image)
                            result_image = cv2.imencode('.png', result_image)[1].tobytes()
                            window['-IMAGE1-'].update(data = result_image)
                            window['text2'].update('Result Sample Image')
                        i = i + 1
                        
                    elif firstflag == True:
                        result_image = make_collage(im_list, default_size_of_image)
                        result_image = cv2.imencode('.png', result_image)[1].tobytes()
                        window['-IMAGE1-'].update(data = result_image)
                        window['text2'].update('Result Sample Images')
                        firstflag = 0
                    
                    process = sg.one_line_progress_meter("Progress", i, images_to_generate, keep_on_top = True, no_titlebar = True, orientation = 'h')    
                    if not process:
                        cancelflag = 1
                        sg.Popup('Augmentation Stopped by user!\nCheck your folder to see the output.\nYou can now exit safely or start another round.', keep_on_top = True, no_titlebar = True)
                        break
                    
                if not cancelflag:
                    sg.Popup('Augmentation Finished Successfully!\nCheck your folder to see the output.\nYou can now exit safely or start another round.', keep_on_top = True, no_titlebar = True)

            else:
                sg.Popup('Please open a valid folder!', keep_on_top = True, no_titlebar = True)

        elif event == 'Check_All':
            window["check Original"].update(True)
            window["check Gray"].update(True)
            window["check Zoom"].update(True)
            window["check Brightness"].update(True)
            window["check Horizontal Flip"].update(True)
            window["check Vertical Flip"].update(True)
            window["check Horizontal Shift"].update(True)
            window["check Vertical Shift"].update(True)
            window["check Rotation"].update(True)
            window['Uncheck_All'].update(False)

        elif event == 'Uncheck_All':
            window["check Original"].update(False)
            window["check Gray"].update(False)
            window["check Zoom"].update(False)
            window["check Brightness"].update(False)
            window["check Horizontal Flip"].update(False)
            window["check Vertical Flip"].update(False)
            window["check Horizontal Shift"].update(False)
            window["check Vertical Shift"].update(False)
            window["check Rotation"].update(False)
            window['Check_All'].update(False)

        elif event.startswith('check'):
            if not values[event]:
                window['Check_All'].update(False)
            else:
                window['Uncheck_All'].update(False)

        elif event == 'Show Folder':
            if augmented_path and os.path.isdir(augmented_path):
                os.startfile(augmented_path)
            else:
                sg.Popup('Please open a valid folder!', keep_on_top = True, no_titlebar = True)

    window.close()

def main():
    user_interface()

if __name__ == "__main__":
    main()
