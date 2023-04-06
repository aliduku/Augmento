import os
import cv2
import numpy as np
import PySimpleGUI as sg

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

def zoom(img):
    value = np.random.uniform(0.7, 1)
    h, w = img.shape[:2]
    h_new = int(value*h)
    w_new = int(value*w)
    h_start = np.random.randint(0, h-h_new)
    w_start = np.random.randint(0, w-w_new)
    return img[h_start: h_start + h_new, w_start:w_start + w_new, :]

def rotation(img):
    angle = np.random.uniform(-90, 90)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((int(w/2), int(h/2)), angle, 1)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))

def brightness(img):
    value = np.random.uniform(0, 100)
    return cv2.add(img, value)

def shift(img, axis):
    h, w = img.shape[:2]
    ratio = np.random.uniform(-0.25, 0.25)
    shift_pixels = int(ratio * (h if axis == 0 else w))
    img = np.roll(img, shift_pixels, axis)
    if axis == 0:
        if shift_pixels > 0:
            img[:shift_pixels, :, :] = (255, 255, 255)
        else:
            img[shift_pixels:, :, :] = (255, 255, 255)
    else:
        if shift_pixels > 0:
            img[:, :shift_pixels, :] = (255, 255, 255)
        else:
            img[:, shift_pixels:, :] = (255, 255, 255)

    return img

def flip(img, direction):
    return cv2.flip(img, direction)

def resize_image(original_image, size_of_image):
    width, height = size_of_image
    scale = max(original_image.shape[1]/width, original_image.shape[0]/height)
    w, h = int(original_image.shape[1] /scale), int(original_image.shape[0]/scale)
    original_image = cv2.resize(original_image, (w, h))
    delta_w = abs(width - w)
    delta_h = abs(height - h)
    return cv2.copyMakeBorder(original_image, delta_h//2, delta_h - delta_h //2, delta_w//2, delta_w - delta_w//2, cv2.BORDER_CONSTANT, value=(255, 255, 255))

def save_img(original_image, result_image, size_of_image, image_path, i, mergeflag, images_to_generate, im_list, firstflag, augmented_path):
    if i <= images_to_generate:
        result_image = resize_image(original_image, size_of_image)
        new_image_path = f"{augmented_path}/augmented_image_{i}.jpg"
        cv2.imwrite(new_image_path, result_image)
        if not mergeflag:
            original_image = cv2.imread(image_path)
            i = i + 1
            if firstflag:
                im_list.append(result_image)
    return result_image, original_image, i, im_list

def make_collage(imgs, default_size_of_image):
    n = int(np.ceil(np.sqrt(len(imgs))))
    padimg = np.zeros((default_size_of_image[0], default_size_of_image[1], 3), dtype=np.uint8)
    padmat = [[padimg] * n for _ in range(n)]
    for i in range(min(len(imgs), n**2)):
        img = cv2.resize(imgs[i], default_size_of_image)
        padmat[i//n][i % n] = img
    imgs_2d = cv2.vconcat([cv2.hconcat(row) for row in padmat])
    imgs_2d = cv2.resize(imgs_2d, default_size_of_image)
    return imgs_2d

def show_popup(msg):
    sg.Popup(msg, keep_on_top=True, no_titlebar=True)

def user_interface():
    foldername, augmented_path = None, None
    sg.theme('DarkBlack')
    default_size_of_image = (400, 400)
    augmentations = [('check Original', lambda img: img), ('check Gray', grayscale), ('check Zoom', zoom), ('check Rotation', rotation), ('check Brightness', brightness),
                     ('check Horizontal Shift', lambda img: shift(img, 1)), ('check Vertical Shift', lambda img: shift(img, 0)), ('check Horizontal Flip', lambda img: flip(img, 1)),
                     ('check Vertical Flip', lambda img: flip(img, 0)), ]
    augmentation_checkboxes = [name for name, _ in augmentations]

    left_col = [[sg.Image(background_color='black', size=default_size_of_image, key='-IMAGE-', expand_x=True, expand_y=True)],
                [sg.Text('', font=("Defualt", 16), key='text1')]]

    right_col = [[sg.Image(background_color='black', size=default_size_of_image, key='-IMAGE1-', expand_x=True, expand_y=True)],
                 [sg.Text('', font=("Defualt", 16), key='text2')]]

    layout = [[sg.Button('Select Folder', tooltip='Click here to select your dataset folder.'),
               sg.Input(disabled=True, text_color='black', expand_x=True, background_color='white', tooltip='Dataset folder', key='foldername'),
               sg.VerticalSeparator(),
               sg.Text('Enter Number of Output Images'),
               sg.InputText(text_color='black', background_color='white', size=(15, 1), tooltip='Number of images to generate.', key='Number of Output Images'),
               sg.Button('Augment', tooltip='Click here to start the augmentation process.')],
              [sg.HorizontalSeparator()],
              [sg.Checkbox('Merge Augmentations', enable_events=True, tooltip='Check this box to merge all the selected augmentation types.', key='Merge'),
               sg.Checkbox('Check all', enable_events=True, tooltip='Check this box to check all augmentation types.', key='Check_All'),
               sg.Checkbox('Uncheck all', enable_events=True, tooltip='Check this box to uncheck all augmentation types.', key='Uncheck_All'),
               sg.VerticalSeparator(),
               sg.Text('Width'), sg.InputText(text_color='black', background_color='white', size=(10, 1), tooltip='Output image width.', key='Width'),
               sg.Text('Height'), sg.InputText(text_color='black', background_color='white', size=(10, 1), tooltip='Output image height.', key='Height'),
               sg.VerticalSeparator(),
               sg.Button('Show Folder', disabled=True, tooltip='Open output folder.', key='Show Folder')],
              [sg.HorizontalSeparator()],
              [sg.Checkbox('Original', enable_events=True, tooltip='Check this box to return the image unchanged.', key='check Original'),
               sg.Checkbox('Gray', enable_events=True, tooltip='Check this box to return the image in grayscale.', key='check Gray'),
               sg.Checkbox('Zoom', enable_events=True, tooltip='Check this box to return the image zoomed.', key='check Zoom'),
               sg.Checkbox('Rotation', enable_events=True, tooltip='Check this box to return the image rotated.', key='check Rotation'),
               sg.Checkbox('Brightness', enable_events=True, tooltip='Check this box to return the image with different brightness.', key='check Brightness'),
               sg.Checkbox('Horizontal Shift', enable_events=True, tooltip='Check this box to return the image shifted horizontally.', key='check Horizontal Shift'),
               sg.Checkbox('Vertical Shift', enable_events=True, tooltip='Check this box to return the image shifted vertically.', key='check Vertical Shift'),
               sg.Checkbox('Horizontal Flip', enable_events=True, tooltip='Check this box to return the image flipped horizontally.', key='check Horizontal Flip'),
               sg.Checkbox('Vertical Flip', enable_events=True, tooltip='Check this box to return the image flipped vertically.', key='check Vertical Flip')],
              [sg.HorizontalSeparator()],
              [sg.Column(left_col, vertical_alignment='center', justification='center',  k='-C-', element_justification='c'),
               sg.VSeperator(), sg.Column(right_col, vertical_alignment='center', justification='center',  k='-C-', element_justification='c')]]

    window = sg.Window("Augmento", layout, size=(900, 600), icon='icon.ico', enable_close_attempted_event=True, finalize=True, element_justification='c')

    while True:
        event, values = window.read()
        if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, "Exit"):
            break

        elif event == 'Select Folder':
            foldername = sg.popup_get_folder("Please select your dataset folder.", keep_on_top=True, no_titlebar=True, no_window=True)
            window['foldername'].update(foldername)
            if foldername and os.path.isdir(foldername):
                original_path = foldername
                augmented_path = foldername + "/results"
                window['Show Folder'].update(disabled=True)
                images = [os.path.join(original_path, im) for im in os.listdir(original_path) if im.lower().endswith(('.png', '.jpg', '.jpeg'))]

        elif event == 'Augment':
            if foldername and os.path.isdir(foldername) and images:
                selected_augmentations = [name for name in augmentation_checkboxes if values.get(name)]
                if not selected_augmentations:
                    show_popup('Please check at least one checkbox!.')
                    continue

                if augmented_path and os.path.isdir(augmented_path):
                    for file in os.listdir(augmented_path):
                        os.remove(os.path.join(augmented_path, file))
                else:
                    os.mkdir(augmented_path)

                try:
                    images_to_generate = int(values['Number of Output Images'])
                    if images_to_generate <= 0:
                        show_popup('Number of output images should be greater than zero.')
                        continue
                except ValueError:
                    show_popup('Invalid integer for number of output images.\nDefault of 100 will be used.')
                    images_to_generate = 100
                    window['Number of Output Images'].update(images_to_generate)

                try:
                    size_of_image = (int(values['Width']), int(values['Height']))
                except ValueError:
                    show_popup('Invalid integers for width and height.\nDefault of 400 x 400 will be used.')
                    size_of_image = default_size_of_image
                    window['Width'].update(size_of_image[0])
                    window['Height'].update(size_of_image[1])

                start = sg.popup_yes_no('The augmentation process is about to begin!\n(' + str(len(images)) + ') images as input.\n(' + str(images_to_generate) +
                                        ') images as output.', keep_on_top=True, no_titlebar=True)

                if start == 'No':
                    continue
        
                i = 1
                im_list = []
                mergeflag = values["Merge"]
                window['Show Folder'].update(disabled=False)
                
                while i <= images_to_generate:
                    image_path = np.random.choice(images)
                    original_image = cv2.imread(image_path)
                    cancelflag = False
                    firstflag = False

                    if i == 1:
                        result_image = resize_image(original_image, default_size_of_image)
                        result_image = cv2.imencode('.png', result_image)[1].tobytes()
                        window['-IMAGE-'].update(data=result_image)
                        window['text1'].update('Original Sample Image')
                        firstflag = True

                    for augmentation, func in augmentations:
                        if values[augmentation]:
                            original_image = func(original_image)
                            result_image, original_image, i, im_list = save_img(original_image, result_image, size_of_image, image_path, i, mergeflag,
                                                                                images_to_generate, im_list, firstflag, augmented_path)

                    if mergeflag:
                        if firstflag:
                            result_image = resize_image(result_image, default_size_of_image)
                            result_image = cv2.imencode('.png', result_image)[1].tobytes()
                            window['-IMAGE1-'].update(data=result_image)
                            window['text2'].update('Result Sample Image')
                        i += 1

                    elif firstflag:
                        result_image = make_collage(im_list, default_size_of_image)
                        result_image = cv2.imencode('.png', result_image)[1].tobytes()
                        window['-IMAGE1-'].update(data=result_image)
                        window['text2'].update('Result Sample Images')
                        firstflag = False

                    process = sg.one_line_progress_meter("Progress", i, images_to_generate, keep_on_top=True, no_titlebar=True, orientation='h')
                    if not process:
                        cancelflag = 1
                        show_popup('Augmentation Stopped by user!\nCheck your folder to see the output.\nYou can now exit safely or start another round.')
                        break

                if not cancelflag:
                    show_popup('Augmentation Finished Successfully!\nCheck your folder to see the output.\nYou can now exit safely or start another round.')

            else:
                show_popup('Please open a valid folder!')

        elif event == 'Check_All':
            for checkbox_key in augmentation_checkboxes:
                window[checkbox_key].update(value=True)
            window['Uncheck_All'].update(False)

        elif event == 'Uncheck_All':
            for checkbox_key in augmentation_checkboxes:
                window[checkbox_key].update(value=False)
            window['Check_All'].update(False)

        elif event.startswith('check'):
            if not values[event]:
                window['Check_All'].update(False)
            else:
                window['Uncheck_All'].update(False)

        elif event == 'Show Folder':
            os.startfile(augmented_path)

    window.close()

def main():
    user_interface()

if __name__ == "__main__":
    main()