import numpy as np

print('Loading data...')


class cmb_profile:
    def __init__(self, train_path, train_target_path, test_path, test_target_path , num_para=9):
        self.train_path = train_path
        self.train_target_path = train_target_path
        self.test_path = test_path
        self.test_target_path = test_target_path
        self.num_para = num_para
        # self.test_split = test_split

    def open_data(self):
        #        with open(self.data_path) as json_data:
        #            self.allData = json.load(json_data)
        #
        #        with open(self.para_path) as json_data:
        #            self.allPara = json.load(json_data)

        self.trainData = np.load(self.train_path)
        self.trainPara = np.load(self.train_target_path)

        self.testData = np.load(self.test_path)
        self.testPara = np.load(self.test_target_path)

        return self.trainData, self.trainPara, self.testData, self.testPara

    def load_data(self):  # randomize and split into train and test data

        trainData, trainPara, testData, testPara = self.open_data()
        # num_train = int((1 - self.test_split) * num_files)

        num_train = len(trainData)
        np.random.seed(1234)
        shuffleOrder = np.arange(num_train)
        np.random.shuffle(shuffleOrder)
        self.x_train = trainData[shuffleOrder]
        self.y_train = trainPara[shuffleOrder]
        print('training data:', self.x_train.shape)

        num_test = len(testData)
        np.random.seed(123)
        shuffleOrder = np.arange(num_test)
        np.random.shuffle(shuffleOrder)
        self.x_test = testData[shuffleOrder]
        self.y_test = testPara[shuffleOrder]
        print('validation data:', self.y_test.shape)

        return (self.x_train, self.y_train), (self.x_test, self.y_test)


print('Data loaded...')

