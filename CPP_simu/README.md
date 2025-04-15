C代码工程的仿真工程，在linux环境下执行，
支持客户自己训练模型、量化模型，然后使用C可执行程序做测试。
执行方式如：./kws_test <file_path> <threshold> <model.h> 
file_path: 测试的wav文件路径
threshold：模型测试唤醒使用的门限
model.h：模型文件nn_kws_model.hpp