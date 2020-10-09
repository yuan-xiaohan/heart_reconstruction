#!/usr/bin/env python
#-*- coding:UTF-8 -*-（添加）
import os
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def obj_read(obj_path):
    # read obj file and get vertices and faces
    with open(obj_path) as file:
        vertices = []
        faces = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                vertices.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "vt":
                break
            if strs[0] == "f":
                faces.append(
                    (int(strs[1].split('//')[0]), int(strs[2].split('//')[0]), int(strs[3].split('//')[0])))
    vertices = np.array(vertices)  # in matrix form
    faces = np.array(faces)
    return vertices, faces


def obj_write(obj_path, vertices, faces):
    with open(obj_path, "w") as file:
        file.write("# " + str(vertices.shape[0]) + " vertices, " + str(faces.shape[0]) + " faces" + "\n")
        # write vertices
        for i in range(vertices.shape[0]):
            file.write("v ")
            file.write(str(float("{0:.6g}".format(vertices[i, 0]))) + " ")
            file.write(str(float("{0:.6g}".format(vertices[i, 1]))) + " ")
            file.write(str(float("{0:.6g}".format(vertices[i, 2]))) + " ")
            file.write("\n")
        # write faces
        for i in range(faces.shape[0]):
            file.write("f ")
            file.write(str(int(faces[i, 0])) + " ")
            file.write(str(int(faces[i, 1])) + " ")
            file.write(str(int(faces[i, 2])) + " ")
            file.write("\n")


def PCA_operation(Input_dir_obj = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\train\obj"):
    """
    Output：
    # X_aver: (v_num, 3)
    # X_new: (v_num*3, k)
    # VT: (k, n)
    # X_pca_ratio: (k, )
    """
    obj_list = os.listdir(Input_dir_obj)

    # get vertices and faces
    X = []
    for obj in obj_list:
        print(obj)
        obj_path = os.path.join(Input_dir_obj, obj)
        vertices, faces = obj_read(obj_path)
        X.append(vertices)

    X = np.array(X)  # (m, v_num, 3)


    # get mean obj
    X_aver = np.mean(X, 0)  # (v_num, 3)
    X = X - np.repeat(np.expand_dims(X_aver, axis=0), X.shape[0], axis=0)
    '''
    print("m.shape", X.shape)
    print("aver.shape", X_aver.shape)
    np.savetxt(os.path.join(output_dir, "aver.txt"), X_aver, fmt='%0.6f')
    # write aver obj
    obj_write(os.path.join(output_dir, "aver.obj"), X_aver, faces)
    '''

    X_reshape = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))  # (m, v_num*3)
    X_reshape = X_reshape.transpose(1, 0)  # (v_num*3, m)
    # print(X_reshape.shape)

    # PCA start
    pca = PCA(n_components=0.99)  # Return the data after PCA, n_com ponents可以是降维结果的维数或者百分比
    X_new = pca.fit_transform(X_reshape)  # (v_num*3, k)
    VT = pca.components_  # vh，[k, v_num]
    '''
    print("X_new.shape", X_new.shape)
    print(X_new)
    print("vh.shape", pca.components_.shape)  # vh，[k, v_num]
    # output to txt
    np.savetxt(os.path.join(output_dir, "pca_coefficient_all.txt"), pca.components_, fmt='%0.6f')
    '''

    # The percentage of variance of each of the remaining k components
    X_pca_ratio = pca.explained_variance_ratio_


    return X_aver, X_new, VT, X_pca_ratio


def reconstruction(output, X_new, aver):
    """
    # output: (m, k)
    # VT: (k, n)
    # X_new: (v_num*3, k)
    # aver: (v_num, 3)
    # m_reconstruct:(m, v_num, 3)
    """
    m_reconstruct = np.dot(X_new, output.T)  # X = X_new * VT
    print(m_reconstruct.shape)
    m_reconstruct = m_reconstruct.transpose(1, 0)  # (1, v_num*3)
    m_reconstruct = np.repeat(np.expand_dims(aver, axis=0), output.shape[0], axis=0) + np.reshape(m_reconstruct,
                                                                                                 [output.shape[0], aver.shape[0], aver.shape[1]])
    return m_reconstruct


def main():
    # start
    Input_dir_obj = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\train\obj"
    output_dir = r"F:\XiaohanYuan_Data\heart_reconstruction\3D"


    obj_list = os.listdir(Input_dir_obj)
    m = []
    for obj in obj_list:
        print(obj)
        obj_path = os.path.join(Input_dir_obj, obj)
        vertices, faces = obj_read(obj_path)
        # print(vertices.shape, faces.shape)
        m.append(vertices)

    m = np.array(m)  # (m, v_num, 3)
    # print(m.shape)

    # mean obj
    m_aver = np.mean(m, 0)  # (v_num, 3)
    m = m - np.repeat(np.expand_dims(m_aver, axis=0), m.shape[0], axis=0)
    vertices_num = m.shape[1]
    print("m.shape", m.shape)
    print("aver.shape", m_aver.shape)
    # 保存平均模型
    # np.savetxt(os.path.join(output_dir, "aver.txt"), m_aver, fmt='%0.6f')

    '''
    # write aver obj
    obj_out = "aver.obj"
    obj_write(obj_out, m_aver, faces)
    '''

    m_reshape = np.reshape(m, (m.shape[0], m.shape[1]*m.shape[2]))  # (m, v_num*3)
    m_reshape = m_reshape.transpose(1, 0)  # (v_num*3, m)
    print(m_reshape.shape)

    # PCA
    pca = PCA(n_components=0.99)  # Return the data after PCA, n_com ponents可以是降维结果的维数或者百分比
    m_reduced = pca.fit_transform(m_reshape)  # (v_num*3, k)
    print("m_reduced.shape", m_reduced.shape)
    # print(m_reduced)
    # pca.components_.T[:, 0]  # Each principal component
    print("vh.shape", pca.components_.shape)  # vh，[k, v_num]
    # print("mean", pca.mean_)

    # output to txt
    # np.savetxt(os.path.join(output_dir, "pca_coefficient_all.txt"), pca.components_, fmt='%0.6f')

    print(type(pca.explained_variance_ratio_))  # The percentage of variance of each of the remaining k components


    '''
    #  make train and val pca coefficient
    pca_dir = r"F:\XiaohanYuan_Data\heart_reconstruction\3D\train\pca_coefficient.txt"
    pca = np.loadtxt(pca_dir, dtype=np.float)
    pca_test = pca.T[0]
    print(pca.shape)
    train = pca[:, :131]  # :train_num
    val = pca[:, 131:]  # train_num:
    print(train.shape, val.shape)
    np.savetxt(r'F:\XiaohanYuan_Data\heart_reconstruction\3D\train\pca_coefficient.txt', train, fmt='%0.6f')
    np.savetxt(r'F:\XiaohanYuan_Data\heart_reconstruction\3D\val\pca_coefficient.txt', val, fmt='%0.6f')
    '''

    '''
    # Observe what the principal components look like
    obj_write("main_component_0.obj", m_aver + np.reshape(m_reduced[:, 0], [vertices_num, 3]), faces)
    obj_write("main_component_1.obj", m_aver + np.reshape(m_reduced[:, 1], [vertices_num, 3]), faces)
    obj_write("main_component_2.obj", m_aver + np.reshape(m_reduced[:, 2], [vertices_num, 3]), faces)
    obj_write("main_component_3.obj", m_aver + np.reshape(m_reduced[:, 3], [vertices_num, 3]), faces)


    obj_write("main_component_0_n.obj", np.reshape(m_reduced[:, 0], [vertices_num, 3]), faces)
    obj_write("main_component_1_n.obj", np.reshape(m_reduced[:, 1], [vertices_num, 3]), faces)
    obj_write("main_component_2_n.obj", np.reshape(m_reduced[:, 2], [vertices_num, 3]), faces)
    obj_write("main_component_3_n.obj", np.reshape(m_reduced[:, 3], [vertices_num, 3]), faces)
    '''



    # Convert the reduced data into original data
    ''' 
    # 直接使用pca模块，但是又多加了一个mean
    m_reconstruct = pca.inverse_transform(m_reduced)  # (v_num*3, m)
    m_reconstruct = m_reconstruct.transpose(1, 0)  # (m, v_num*3)
    m_reconstruct = np.repeat(np.expand_dims(m_aver, axis=0), m.shape[0], axis=0) + np.reshape(m_reconstruct, m.shape)
    print("m_reconstruct", m_reconstruct.shape)  # (m, v_num, 3)
    obj_write("reconstruct_template.obj", m_reconstruct[0, :, :], faces)
    

    # 自己写的
    m_reconstruct = np.dot(m_reduced, pca.components_)  # (v_num*3, m)
    m_reconstruct = m_reconstruct.transpose(1, 0)  # (m, v_num*3)
    m_reconstruct = np.repeat(np.expand_dims(m_aver, axis=0), pca.components_.shape[1], axis=0) + \
                    np.reshape(m_reconstruct, [pca.components_.shape[1], m_aver.shape[0], m_aver.shape[1]])
    print("m_reconstruct", m_reconstruct.shape)  # (m, v_num, 3)
    obj_write("reconstruct_template.obj", m_reconstruct[0, :, :], faces)
    '''


'''
class OBJ:
    def __init__(self, filename):
        self.vertices = []
        self.faces = []

        with open(obj_path) as file:
            while 1:
                line = file.readline()
                if not line:
                    break
                strs = line.split(" ")
                if strs[0] == "v":
                    self.vertices.append((float(strs[1]), float(strs[2]), float(strs[3])))
                if strs[0] == "vt":
                    break
                if strs[0] == "f":
                    self.faces.append(
                        (float(strs[1].split('//')[0]), float(strs[2].split('//')[0]), float(strs[3].split('//')[0])))
        # in matrix form
        self.vertices = np.array(self.vertices)
        self.faces = np.array(self.faces)
        # print(vertices.shape, faces.shape)
        m.append(self.vertices)
'''

if __name__ == "__main__":
    main()




