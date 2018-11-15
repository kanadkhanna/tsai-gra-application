"""
Author: Kanad Khanna
Date:   11/13/2018
Description:
        Saves tf_records of the training images.

        This is the first file of an easier-to-edit, more effective script that
        uses the Tensorflow backend and all of its data augmentation options.
"""

import os
import pandas as pd
from PIL import Image
import tensorflow as tf
from object_detection.utils import dataset_util




def load_ground_truth():
    """
    Return DataFrame containing ground truth labels.
    """
    # Read file
    data = pd.read_csv('images/train/gt.txt', sep=';', header=None)
    data.columns = ['Filename', 'Left', 'Top', 'Right', 'Bottom', 'ClassID']

    # Limit to prohibitory signs (circular, white ground, red border)
    data = data.query('ClassID in [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]')
    
    return data



def create_tf_example(example):
    """
    Args:
        dict(
            filename: '{}.jpg',
            image: PIL object containing JPEG-encoded image,
            gt: DataFrame with ground truth labels for the image
        )

    Returns:
        tf record
    """
    # Basic image info
    image = example['image'] # PIL object
    filename = example['filename'].encode(encoding='utf-8') # Filename of the image
    width, height = image.size # Image height
    encoded_image_data = image.tobytes() # Encoded image bytes
    image_format = b'jpeg' # b'jpeg' or b'png'

    # Initialize features
    xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = [] # List of string class name of bounding box (1 per box)
    classes = [] # List of integer class id of bounding box (1 per box)

    # Populate using ground truth
    gt = example['gt']
    xmins = (gt['Left'] / float(width)).tolist()
    xmaxs = (gt['Right'] / float(width)).tolist()
    ymins = (gt['Top'] / float(height)).tolist()
    ymaxs = (gt['Bottom'] / float(height)).tolist()
    classes_text = [b'round_sign' for i in range(len(gt))]
    classes = [1 for i in range(len(gt))]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example



def main(_):
    # Load ground truth labels
    gt = load_ground_truth() # DataFrame containing ground truth labels

    # Create a TFWriter
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    # Make a list of examples
    examples = []
    rootdir = 'images/train'

    print()
    print('Converting images to JPEG...')
    
    for file in os.listdir(rootdir):
        # Get filename/extension
        filename, file_extension = os.path.splitext(file)
        
        if file_extension == '.ppm':
            # Open and save file as jpeg if not already done
            if not os.path.exists('images/train_jpg/{}.jpg'.format(filename)):
                fp = '/'.join((rootdir, file))
                Image.open(fp).save(
                    'images/train_jpg/{}.jpg'.format(filename), 'JPEG')
                print('Saved {}.jpg'.format(filename))

            # Read jpeg. Append filename, PIL object, and g.t. labels to list
            im = Image.open('images/train_jpg/{}.jpg'.format(filename))
            examples.append({
                'filename': filename + '.jpg',
                'image': im,
                'gt': gt.query('Filename == "{}"'.format(file))
            })
    print('Conversion complete.')


    # Write records
    print('Creating TFRecords...')
    for example in examples:
        tf_example = create_tf_example(example)
        writer.write(tf_example.SerializeToString())
        print('\tCreated record for {}'.format(example['filename']))

    writer.close()




if __name__ == '__main__':
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    output_path = os.path.abspath('./training_data.record')

    flags = tf.app.flags
    flags.DEFINE_string('output_path', output_path, '')
    FLAGS = flags.FLAGS

    tf.app.run()