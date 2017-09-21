import luigi
import time
import os
import numpy as np
import tensorflow as tf

from build_features import int32_to_uint8, threshold_confidence
from PIL import Image

""" The preprocessing data pipeline

The pipeline is composed of the following steps:

1- DatasetToRawCSV makes a CSV file with all the absolute paths of the images
2- MakeGroundtruths constructs a confidence map for each file.

"""

data_folder='/Users/afusco/Developer/tesi/data'

class RawFiles(luigi.ExternalTask):
    """ DAG placeholder for input raw directory """
    dataset_name = luigi.Parameter()

    def output(self):
        raw_path = os.path.join(data_folder, 'raw', self.dataset_name)
        if not os.path.isdir(raw_path):
            raise ValueError("Invalid dataset raw input " + self.dataset_name)

        return luigi.LocalTarget(raw_path)

class DatasetToRawCSV(luigi.Task):
    """ Compiles a csv list of raw tuples """
    dataset_name = luigi.Parameter()

    def requires(self):
        return RawFiles(self.dataset_name)

    def output(self):
        return luigi.LocalTarget(data_folder + '/interim/' + self.dataset_name + '/raw.csv')

    def run(self):
        raw_dir = self.input().path
        dirs = [ 'left', 'disparity', 'gt' ]

        # Get only the filenames that are in all directories
        filenames = [f for f in os.listdir(os.path.join(raw_dir, dirs[0]))
                if all(os.path.isfile(os.path.join(raw_dir, data_dir, f)) for data_dir in dirs)]

        with self.output().open('w') as outfile:
            for f in filenames:
                line = os.path.join(raw_dir, dirs[0], f)
                for d in dirs[1:]:
                    line = line + ',' + os.path.join(raw_dir, d, f)
                    
                outfile.write(line + '\n')

class MakeAllConfidenceMaps(luigi.Task):
    dataset_name = luigi.Parameter()

    def requires(self):
        return DatasetToRawCSV(self.dataset_name)

    def output(self):
        return luigi.LocalTarget(data_folder + '/interim/' + self.dataset_name + '/processed.csv')

    def run(self):

        image_paths = []

        conf_folder = data_folder + '/interim/' + self.dataset_name + '/conf/'

        if not os.path.exists(conf_folder):
            #todo exception
            os.mkdir(conf_folder)
        
        with self.input().open() as incsv:
            for line in incsv:
                split = [x.strip() for x in line.split(',')]
                
                image_paths.append({
                    'left': split[0],
                    'disp': split[1],
                    'gt': split[2],
                    'conf':  conf_folder + os.path.basename(split[0]) + '.npy'
                })

        for path_tuple in image_paths:
            

            d_raw = np.array(Image.open(path_tuple['disp']))
            g_raw = np.array(Image.open(path_tuple['gt']))

            ### maybe bug?
            g_raw = int32_to_uint8(g_raw)
            confidence = threshold_confidence(d_raw, g_raw, 2)

            np.save(path_tuple['conf'], confidence)

        # Make CSV
        with self.output().open('w') as outfile:
            for path_tuple in image_paths:
                line = path_tuple['left'] + ',' + path_tuple['disp'] + ',' + path_tuple['conf']
                outfile.write(line + '\n')


class MakeTFRecords(luigi.Task):
    number_of_splits = luigi.IntParameter(default=5)
    dataset_name = luigi.Parameter()

    def requires(self):
        return MakeAllConfidenceMaps(self.dataset_name)

    def output(self):
        return luigi.LocalTarget(data_folder + '/processed/' + self.dataset_name + '/processed.csv')

    def run(self):
        lines = self.input().open().readlines()
        samples_count = len(lines)
        counter = 0

        #TODO refactor this out
        # Define a raw feature
        def _bytes_feature(value):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        # Define an integer feature
        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


        outfile = self.output().open('w')
        for split in range(self.number_of_splits):

            tfrecords_path = data_folder + '/processed/' + self.dataset_name + '/fold-{}.tfrecords'.format(split+1)
            writer = tf.python_io.TFRecordWriter(tfrecords_path)

            #Get lines bound for splits
            low_bound = int(split * samples_count/self.number_of_splits)
            high_bound = int(low_bound + samples_count/self.number_of_splits)

            if high_bound > samples_count:
                high_bound = samples_count


            for line in lines[low_bound:high_bound]:
                counter += 1

                #get paths
                left_path, disp_path, conf_path = [x.strip() for x in line.split(',')]

                #load images
                left = np.array(Image.open(left_path))
                disp = np.array(Image.open(disp_path))
                conf = np.load(conf_path)


                # We make sure that height and width match
                # for all the three images
                if (left.shape[0] != disp.shape[0] or left.shape[1] != disp.shape[1] or
                    left.shape[0] != conf.shape[0] or left.shape[1] != conf.shape[1]):
                    print("Errore per l'immagine", left_path)
                    raise Exception("L'immagine numero", counter, "in", left_path,
                                    "è associata a mappe di dimensioni differenti")

                # The reason to store image sizes was demonstrated
                # in the previous example -- we have to know sizes
                # of images to later read raw serialized string,
                # convert to 1d array and convert to respective
                # shape that image used to have.
                height = left.shape[0]
                width = left.shape[1]

                # convert to binary
                left_raw = left.tostring()
                disp_raw = disp.tostring()
                conf_raw = conf.tostring()

                #create sample
                example = tf.train.Example(features=tf.train.Features(feature={
                    'height': _int64_feature(height),
                    'width': _int64_feature(width),
                    'left_raw': _bytes_feature(left_raw),
                    'disp_raw': _bytes_feature(disp_raw),
                    'conf_raw': _bytes_feature(conf_raw)}))

                writer.write(example.SerializeToString())
                if counter % 20 == 0:
                    print("Written", counter, "out of", samples_count)


            writer.close()
            outfile.write(tfrecords_path)

        outfile.close()


if __name__ == '__main__':
    luigi.run()
