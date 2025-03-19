import numpy as np
class Cross_Validation:

    def __init__(self, data, label):
        self.data = data
        self.label = label
 
    def data_shuffle(self):
        index = list(range(len(self.data)))
        np.random.shuffle(index)
        shuffle_data = np.zeros(shape=self.data.shape)
        shuffle_label = np.zeros(shape=self.label.shape)
        for i, j in enumerate(index):
            shuffle_data[i,:,:] = self.data[j,:,:]
            shuffle_label[i, :] = self.label[j, :]
        return shuffle_data, shuffle_label

    def find_index(self, target_label, label):
        index_list = []
        for i in range(len(label)):
            if np.argmax(label[i,:])==target_label:
                index_list.append(i)
        return index_list
    
    def test_data_generation(self):
        data_list = []
        label_list = []
        sample_size = 1000
    
        shuffle_data, shuffle_label = self.data_shuffle()
        for i in range(9):
            target_index = self.find_index(target_label=i, label=shuffle_label)
            print(len(target_index),i)
            target_data = np.zeros(shape=(sample_size, shuffle_data.shape[1], shuffle_data.shape[2]))
            target_label = np.zeros(shape=(sample_size, shuffle_label.shape[1]))
            
            for j in range(sample_size):
                target_data[j,:,:] = shuffle_data[target_index[j], :, :]
                target_label[j,:] = shuffle_label[target_index[j], :]
    
            data_list.append(target_data)
            label_list.append(target_label)
    
        return data_list, label_list
