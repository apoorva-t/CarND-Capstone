from styx_msgs.msg import TrafficLight
import tensorflow as tf
import os
import rospy
import numpy as np
import operator

# Code in this class is adapted from tensorflow object_detection/object_detection_tutorial.ipynb

min_score_thresh = .5

class TLClassifier(object):
    def __init__(self, isSite):
        #TODO load classifier
        self.light_state = TrafficLight.UNKNOWN
        self.category_dict = {1: "Green", 2: "Yellow", 3: "Red", 4: "off"}
        dirpath = os.getcwd()

        if isSite:
            PATH_TO_FROZEN_GRAPH = dirpath + '/exported_graphs_site' + '/frozen_inference_graph.pb'
            rospy.loginfo("Model for site")
        else:
            PATH_TO_FROZEN_GRAPH = dirpath + '/exported_graphs' + '/frozen_inference_graph.pb'
            rospy.loginfo("Model for simulator")

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                self.tensor_dict = {}
                for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
                self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        #self.light_state = TrafficLight.UNKNOWN  # use previous light_state if current is unknown
        #image_np = self.load_image_into_numpy_array(image)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Run inference
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                output_dict = sess.run(self.tensor_dict, feed_dict={self.image_tensor: image_np_expanded})
                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                #if 'detection_masks' in output_dict:
                #    output_dict['detection_masks'] = output_dict['detection_masks'][0]

                '''
                for i in range(output_dict['detection_boxes'].shape[0]):
                    if output_dict['detection_scores'] is None or output_dict['detection_scores'][i] > min_score_thresh:
                        class_name = 'Unknown'
                        if output_dict['detection_classes'][i] in self.category_dict.keys():
                            class_name = self.category_dict[output_dict['detection_classes'][i]]

                        if class_name == 'Green':
                            self.light_state = TrafficLight.GREEN
                        elif class_name == 'Red':
                            self.light_state = TrafficLight.RED
                        elif class_name == 'Yellow':
                            self.light_state = TrafficLight.YELLOW

                        rospy.loginfo("Traffic light state from classifier: %s", class_name)
                '''
                scores = output_dict['detection_scores']
                if output_dict['detection_boxes'].shape[0]:
                    if scores is None:
                        class_name = 'Unknown'
                        #rospy.loginfo("Traffic light state from classifier: %s", class_name)
                    else:
                        max_index = np.argmax(scores)
                        if scores[max_index] > min_score_thresh:
                            if output_dict['detection_classes'][max_index] in self.category_dict.keys():
                                class_name = self.category_dict[output_dict['detection_classes'][max_index]]

                            if class_name == 'Green':
                                self.light_state = TrafficLight.GREEN
                            elif class_name == 'Red':
                                self.light_state = TrafficLight.RED
                            elif class_name == 'Yellow':
                                self.light_state = TrafficLight.YELLOW

                            #rospy.loginfo("Traffic light state from classifier: %s, score: %f", class_name, scores[max_index])


        return self.light_state
        #return TrafficLight.UNKNOWN
