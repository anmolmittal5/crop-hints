import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser
import time
import logging



class Detection(object):
    def __init__(self):
        parser = ConfigParser()
        parser.read("./properties.config")
        self.yolo_weights = parser.get("detection_config", "yolo_weights")
        self.yolo_cfg = parser.get("detection_config", "yolo_cfg")
        self.yolo_labels = parser.get("detection_config", "yolo_labels")
        min_probability = parser.get("detection_config", "min_probability")
        threshold = parser.get("detection_config", "threshold")
        buffer_px = parser.get("detection_config", "buffer_px")
        self.buffer_px = int(buffer_px)
        self.min_probability = float(min_probability)
        self.threshold = float(threshold)

    def _get_labels(self):
        with open(self.yolo_labels) as f:
            labels = [line.strip() for line in f]

        return labels



    def model(self, image_BGR, h, w):
        content_dict = {}
        bounding_boxes = []
        confidences = []
        class_numbers = []
        objects = []


        network = cv2.dnn.readNetFromDarknet(self.yolo_cfg, self.yolo_weights)
        layers_names_output = network.getUnconnectedOutLayersNames()
        labels = self._get_labels()


        # image_BGR = cv2.imread(input_image)
        # h, w = image_BGR.shape[:2]
        # image_ratio = w/h
        blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
        network.setInput(blob)
        output_from_network = network.forward(layers_names_output)
        for results in output_from_network:
            for detected_objects in results:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]
                if confidence_current > self.min_probability:
                    objects.append(labels[class_current])
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.min_probability, self.threshold)
        # new code from here
        # counter = 1
        # if len(results) > 0:
        #     for i in results.flatten():
        #         counter += 1
        #         if objects[i] == 'person':
        #             print(person_bbox)
        #             try:
        #                 x_min, y_min = person_bbox[i][0], person_bbox[i][1]
        #                 box_width, box_height = person_bbox[i][2], person_bbox[i][3]
        #             except:
        #                 pass
        #
        #         else:
        #             x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        #             box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
        #
        #         # minimums
        #         x_mins.append(x_min)
        #         y_mins.append(y_min)
        #         # maximums
        #         x_max = x_min + box_width
        #         y_max = y_min + box_height
        #         x_maxes.append(x_max)
        #         y_maxes.append(y_max)

                # print(objects)
                # detected_objects = objects
        content_dict["bounding_boxes"] = bounding_boxes
        content_dict["confidences"] = confidences
        content_dict["colors"] = colors
        content_dict["class_numbers"] = class_numbers
        content_dict["results"] = results
        content_dict["detected_objects"] = objects
        # content_dict["x_mins"] = x_mins
        # content_dict["y_mins"] = y_mins
        # content_dict["x_maxes"] = x_maxes
        # content_dict["y_maxes"] = y_maxes



        print(objects)
        print("_________________________")


        return content_dict


    def get_crop_hints(self, input_image, aspect_ratio):
        aspect_ratio_str = str(aspect_ratio)
        x_mins = []
        y_mins = []
        x_maxes = []
        y_maxes = []
        crop_points = []
        image_BGR = cv2.imread(input_image)
        logging.info("Successfully Read The Input Image")
        h, w = image_BGR.shape[:2]
        original_image_ratio = w/h
        labels = self._get_labels()
        content_dict = self.model(image_BGR, h, w)
        results = content_dict.get("results")
        bounding_boxes = content_dict.get("bounding_boxes")
        detected_objects = content_dict.get("detected_objects")
        class_numbers = content_dict.get("class_numbers")
        # x_mins = content_dict.get("x_mins")
        # y_mins = content_dict.get("y_mins")
        # x_maxes = content_dict.get("x_maxes")
        # y_maxes = content_dict.get("y_maxes")
        counter = 1
        if len(results) > 0:
            for i in results.flatten():
                counter += 1
                # indices = [index for index, element in enumerate(detected_objects) if element == 'person']

                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
                label = str(labels[class_numbers[i]])
                if label == 'person':
                    logging.info("Person has been detected in the image, therefore adding buffer pixels to the top")
                    y_min -= self.buffer_px
                # minimums
                x_mins.append(x_min)
                y_mins.append(y_min)
                # maximums
                x_max = x_min + box_width
                y_max = y_min + box_height
                x_maxes.append(x_max)
                y_maxes.append(y_max)




                # for index, obj in enumerate(detected_objects):
                #     if obj == 'person':
                #         y_min -= 100
                #         print("updating y_min")
                #         # minimums
                #         x_mins.append(x_min)
                #         y_mins.append(y_min)
                #         # maximums
                #         x_max = x_min + box_width
                #         y_max = y_min + box_height
                #         x_maxes.append(x_max)
                #         y_maxes.append(y_max)
                #
                #     else:
                #
                #         # minimums
                #         x_mins.append(x_min)
                #         y_mins.append(y_min)
                #         # maximums
                #         x_max = x_min + box_width
                #         y_max = y_min + box_height
                #         x_maxes.append(x_max)
                #         y_maxes.append(y_max)

                # color_box_current = colors[class_numbers[i]].tolist()
                # cv2.rectangle(image_BGR, (x_min, y_min), (x_min + box_width, y_min + box_height), color_box_current,
                #               2)
                # text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
                # cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
                #             color_box_current, 3)
                # plt.figure(figsize=(20, 10))
                # plt.imshow(cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB))


        # computing local minima and maxima
        if x_mins != [] or x_maxes != [] or y_mins != [] or y_maxes != []:
            local_min_x = min(x_mins)
            local_min_y = min(y_mins)
            local_max_x = max(x_maxes)
            local_max_y = max(y_maxes)
            if original_image_ratio > 1:
                # landscape image
                logging.info("Processing a Landscape Image")
                local_min_y = 0
                local_max_y = h
                new_width = aspect_ratio * h
                x_right_dist = w - local_max_x
                x_left_dist = local_min_x
                freeze_x = min(x_right_dist, x_left_dist)
                if freeze_x == x_right_dist:
                    local_min_x = local_max_x - new_width
                else:
                    local_max_x = local_min_x + new_width

                crop_points.append(round(local_min_x))
                crop_points.append(round(local_min_y))
                crop_points.append(round(local_max_x))
                crop_points.append(round(local_max_y))



                # cv2.rectangle(image_BGR, (local_min_x - self.buffer_px, local_min_y), (local_max_x + self.buffer_px, local_max_y), (255, 0, 0), 2)

            else:
                # portrait image
                logging.info("Processing a Portrait Image")
                local_min_x = 0
                local_max_x = w
                new_height = w / aspect_ratio
                y_top_dist = local_min_y
                y_bottom_dist = h - local_max_y
                height_dist = y_bottom_dist - y_top_dist

                if 'person' in detected_objects:
                    # local_min_y -= self.buffer_px
                    # if new_height > height_dist:
                    #     diff_px = new_height - height_dist
                    #     local_min_y -= diff_px
                    local_max_y = local_min_y + new_height
                else:
                    freeze_y = min(y_top_dist, y_bottom_dist)
                    if freeze_y == y_top_dist:
                        local_max_y = local_min_y + new_height
                    else:
                        local_min_y = local_max_y - new_height

                crop_points.append(round(local_min_x))
                crop_points.append(round(local_min_y))
                crop_points.append(round(local_max_x))
                crop_points.append(round(local_max_y))

                # new code from here
                # local_min_x = 0
                # local_max_x = w
                # new_height = w / aspect_ratio
                # for obj in detected_objects:
                #     if obj == 'person':
                #         local_min_y = 0
                #         local_max_y = new_height
                #     else:
                #         y_top_dist = local_min_y
                #         y_bottom_dist = h - local_max_y
                #         freeze_y = min(y_top_dist, y_bottom_dist)
                #         if freeze_y == y_top_dist:
                #             local_max_y = local_min_y + new_height
                #         else:
                #             local_min_y = local_max_y - new_height
                #
                # crop_points.append(round(local_min_x))
                # crop_points.append(round(local_min_y))
                # crop_points.append(round(local_max_x))
                # crop_points.append(round(local_max_y))

        cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        print(crop_points)
        logging.info("CROP HINTS:{}".format(crop_points))
        try:
            crop_img = image_BGR[crop_points[1]:crop_points[3], crop_points[0]:crop_points[2]]
            logging.info("Successfully Cropped The Input Image with the Input Aspect Ratio")
            i = 0
            while os.path.exists("results/16by9_v2/sample%s_result_{}.jpg".format(aspect_ratio_str) % i):
                i += 1
            # image_path = "results_{}".format(aspect_ratio)
            cv2.imwrite('results/16by9_v2/sample%s_result_{}.jpg'.format(aspect_ratio_str) % i, crop_img)
            # cv2.imwrite('results/original/sample%s_result_original.jpg' % i, image_BGR)
            logging.info("Successfully Stored The Cropped Image Into The Directory")

        except:
            logging.info("Couldn't Find Relevant Crop Hints")
            pass

        # cv2.rectangle(image_BGR, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # plt.figure(figsize=(20, 10))
        # cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
        # i = 0
        # while os.path.exists("cropped_images/sample%s_result.jpg" % i):
        #     i += 1
        # # image_path = "results_{}".format(aspect_ratio)
        # cv2.imwrite('cropped_images/sample%s_result_{}.jpg'.format(aspect_ratio_str) % i, crop_img)












    # def model(self, input_images):
    #     network = cv2.dnn.readNetFromDarknet(self.yolo_cfg, self.yolo_weights)
    #     layers_names_output = network.getUnconnectedOutLayersNames()
    #     labels = self._get_labels()
    #     start = time.time()
    #     for input_file in input_images:
    #         bounding_boxes = []
    #         confidences = []
    #         class_numbers = []
    #         image_BGR = cv2.imread(input_file)
    #         h, w = image_BGR.shape[:2]
    #         image_ratio = w/h
    #         blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    #         colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')
    #         network.setInput(blob)
    #         output_from_network = network.forward(layers_names_output)
    #         for results in output_from_network:
    #             for detected_objects in results:
    #                 scores = detected_objects[5:]
    #                 class_current = np.argmax(scores)
    #                 confidence_current = scores[class_current]
    #                 if confidence_current > self.min_probability:
    #                     box_current = detected_objects[0:4] * np.array([w, h, w, h])
    #                     x_center, y_center, box_width, box_height = box_current
    #                     x_min = int(x_center - (box_width / 2))
    #                     y_min = int(y_center - (box_height / 2))
    #                     bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
    #                     confidences.append(float(confidence_current))
    #                     class_numbers.append(class_current)
    #
    #         results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, self.min_probability, self.threshold)
    #
    #
    #         counter = 1
    #         x_mins = []
    #         y_mins = []
    #         x_maxes = []
    #         y_maxes = []
    #         if len(results) > 0:
    #             for i in results.flatten():
    #                 objects = labels[int(class_numbers[i])]
    #                 print(objects)
    #
    #                 counter += 1
    #                 x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
    #                 box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
    #
    #                 # minimums
    #                 x_mins.append(x_min)
    #                 y_mins.append(y_min)
    #                 # maximums
    #                 x_max = x_min + box_width
    #                 y_max = y_min + box_height
    #                 x_maxes.append(x_max)
    #                 y_maxes.append(y_max)
    #
    #                 color_box_current = colors[class_numbers[i]].tolist()
    #                 cv2.rectangle(image_BGR, (x_min, y_min), (x_min + box_width, y_min + box_height), color_box_current,
    #                               2)
    #                 text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])], confidences[i])
    #                 cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_COMPLEX, 0.7,
    #                             color_box_current, 3)
    #                 # plt.figure(figsize=(20, 10))
    #                 # plt.imshow(cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB))
    #         # computing local minima and maxima
    #         if x_mins != [] or x_maxes != [] or y_mins != [] or y_maxes != []:
    #             print(x_mins)
    #             print(x_maxes)
    #             local_min_x = min(x_mins)
    #             local_min_y = min(y_mins)
    #             print(type(local_min_x))
    #             print(local_min_x)
    #
    #             local_max_x = max(x_maxes)
    #             local_max_y = max(y_maxes)
    #             if image_ratio > 1:
    #                 # landscape image
    #                 local_min_y = 0
    #                 local_max_y = h
    #                 cv2.rectangle(image_BGR, (local_min_x - self.buffer_px, local_min_y), (local_max_x + self.buffer_px, local_max_y), (255, 0, 0), 2)
    #
    #             else:
    #                 # portrait image
    #                 local_min_x = 0
    #                 local_max_x = w
    #                 cv2.rectangle(image_BGR, (local_min_x, local_min_y - self.buffer_px),
    #                               (local_max_x, local_max_y + self.buffer_px), (255, 0, 0), 2)
    #             plt.figure(figsize=(20, 10))
    #             cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB)
    #             i = 0
    #             while os.path.exists("results/sample%s_result.jpg" % i):
    #                 i += 1
    #             cv2.imwrite('results/sample%s_result.jpg' % i, image_BGR)
                # plt.savefig('results/sample%s_result.jpg' % i)







                    # if len(objects) == 1:
                    #     i = 0
                    #     while os.path.exists("results_set1_kogo_images/sample%s_result.jpg" % i):
                    #         i += 1
                    #
                    #     #             fh = open("sample%s.xml" % i, "w")
                    #     plt.savefig('results_set1_kogo_images/sample%s_result.jpg' % i)
                    #
                    # elif len(objects) > 1:
                    #     j = 0
                    #     while os.path.exists("results_set2_kogo_images/sample%s_result.jpg" % j):
                    #         j += 1
                    #
                    #     #             fh = open("sample%s.xml" % i, "w")
                    #     plt.savefig('results_set2_kogo_images/sample%s_result.jpg' % j)
                    #
                    # elif len(objects) == 0:
                    #     k = 0
                    #     while os.path.exists("results_reconsider_kogo_images/sample%s_result.jpg" % k):
                    #         k += 1
                    #
                    #     #             fh = open("sample%s.xml" % i, "w")
                    #     plt.savefig('results_reconsider_kogo_images/sample%s_result.jpg' % k)





if __name__ == "__main__":
    logging.info("Starting Object Detection")

