import luigi
import time
import os
import numpy as np
import tensorflow as tf

from PIL import Image

""" The preprocessing data pipeline

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

        from features.build_features import int32_to_uint8, threshold_confidence

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

class SplitDataset(luigi.Task):
    dataset_name = luigi.Parameter()
    offset = luigi.IntParameter(default=0)
    lines = luigi.IntParameter(default=0)
    filename = luigi.Parameter()

    def requires(self):
        return MakeAllConfidenceMaps(self.dataset_name)

    def output(self):
        return luigi.LocalTarget(data_folder + '/interim/' + self.dataset_name + '/{}.csv'.format(self.filename))

    def run(self):
        with self.input().open() as infile, self.output().open('w') as outfile:
            samples = infile.readlines()

            low_bound = self.offset
            high_bound = self.offset + self.lines
            if self.lines == 0 or high_bound > len(samples):
                high_bound = len(samples)

            for l in samples[low_bound:high_bound]:
                outfile.write(l)

class MakeTFRecords(luigi.Task):
    dataset_name = luigi.Parameter()
    offset = luigi.IntParameter(default=0)
    lines = luigi.IntParameter(default=0)
    filename = luigi.Parameter()

    def requires(self):
        return SplitDataset(self.dataset_name, self.offset, self.lines, self.filename)

    def output(self):
        return luigi.LocalTarget(data_folder + '/processed/' + self.dataset_name + '/{}.tfrecords'.format(self.filename))

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


        # create output
        with self.output().open('w') as f:
            pass

        with self.input().open() as infile, tf.python_io.TFRecordWriter(self.output().path) as writer:

            for line in infile.readlines():
                counter += 1
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
                    print("Written", counter, "out of", len(lines))

            print("Written", counter, "out of", len(lines))

class MakeTestData(luigi.Task):
    dataset_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(data_folder + '/processed/' + self.dataset_name + '/test_data.tfrecords')


    def requires(self):
        return MakeTFRecords(self.dataset_name, 0, 30, 'test_data')

class MakeEvalData(luigi.Task):
    dataset_name = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(data_folder + '/processed/' + self.dataset_name + '/train_data.tfrecords')

    def requires(self):
        return MakeTFRecords(self.dataset_name, 30, 0, 'train_data')

class ProcessData(luigi.Task):
    dataset_name = luigi.Parameter()

    def requires(self):
        yield MakeTestData(self.dataset_name)
        yield MakeEvalData(self.dataset_name)


if __name__ == '__main__':
    # Path hack.
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.realpath(__file__))+'/..'))
    luigi.run()
