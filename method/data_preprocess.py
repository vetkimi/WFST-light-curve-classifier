# 加载数据
shuffle_data_list = 

# gp 拟合
import sys
import numpy as np
from tqdm import tqdm
sys.path.append("/home/yltang/data/lc_processor/python_code/tools")
from make_dense_light_curve_effective import make_dense_light_curve
make_dense = make_dense_light_curve(light_curve=None)

gp_parameter_list = []
sample_size = len(shuffle_data_list)

print("begain gp fitting...")
lc_list = []
concated_lc_list = []
fit_fail_list = []
with tqdm(total=sample_size*10, desc='Processing...', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
    for sample in range(len(shuffle_data_list)):
        mjd = []
        flux = []
        flux_err = []
        #选取的波段数量
        for i in range(3):
            mjd_ = shuffle_data_list[sample][0][i]
            flux_ = shuffle_data_list[sample][1][i]
            flux_err_ = shuffle_data_list[sample][2][i]
            mjd_ = np.array(mjd_)
            flux_ = np.array(flux_)
            flux_err_ = np.array(flux_err_)
            mjd.append(mjd_)
            flux.append(flux_)
            flux_err.append(flux_err_)

        #光变曲线的格式
        lc = [mjd, flux, flux_err]
        concatnet_lc = make_dense.concatenate_lc(light_curve_info=lc)
        try:
            gp_ = make_dense.fit_gps(concat_lc=concatnet_lc)
            lc_list.append(lc)
            concated_lc_list.append(concatnet_lc)
            gp_parameter_list.append(gp_)
        except:
            fit_fail_list.append(sample)
            print(f"----GP failed to fit this data: index{sample}")
            
        pbar.update(10)
print("gp fitting success!")

#定义插值函数
def interpolation(lc):
    mjd_list = []
    flux_list = []
    flux_err_list = []
    weight = []
    #波段数量
    for band in range(3):
        x_pred = np.linspace(0, max(time), int(max(time)-min(time))+1)
        #x_pred = np.linspace(0, 149, 150)
        x_pred = np.vstack([x_pred,np.ones(x_pred.shape)*central_wave_length[band]])
        pred, pred_var = gp_.predict(fluxes, x_pred.T, return_var=True)
        x_pred_ = x_pred[0,:]
        #print(x_pred_)

        mjd = x_pred_
        flux = pred
        flux_err = np.sqrt(pred_var)
        weight_pre_band = list(np.zeros(shape=(len(mjd),)))
        
        for i, generate_mjd in enumerate(mjd):
            for j, real_mjd in enumerate(lc[0][band]):
                if generate_mjd == real_mjd:
                    #print(generate_mjd, real_mjd)
                    flux[i] = lc[1][band][j]
                    flux_err[i] = lc[2][band][j]
                    weight_pre_band[i] = 1.
                    #print(weight_pre_band[i])
                    break
                    
        #print(weight_pre_band)       
        mjd_list.append(mjd)
        flux_list.append(flux)
        flux_err_list.append(flux_err)
        weight.append(weight_pre_band)

    interpolated_data = [mjd_list, flux_list, flux_err_list]
    return interpolated_data, weight

central_wave_length = [3570.0,4767.0,6215.0,7545.0,8708.0,10040.0]
central_wave_length = np.array(central_wave_length)

#开始插值
print("begain interpolation...")
interpolated_data_list = []
weight_list = []
with tqdm(total=sample_size*10, desc='Processing...', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
    for sample in range(len(lc_list)):
        lc = lc_list[sample]
        concatnet_lc = concated_lc_list[sample]
        gp_ = gp_parameter_list[sample]
        time = concatnet_lc[0]
        fluxes = concatnet_lc[1]
        flux_errs = concatnet_lc[2]
        filters = concatnet_lc[3]
        try:
            interpolated_data, weight = interpolation(lc=lc)
            interpolated_data_list.append(interpolated_data)
            weight_list.append(weight)
        except:
            fit_fail_list.append(sample)
            print(f"----GP failed to fit this data: index{sample}")
        pbar.update(10)
print("interpolation success!")

def trigger(data):
    """
    """
    trigger_list = []
    for index in range(len(data)):
        mjd_ = data[index][0][1]
        flux_ = data[index][1][1]
        flux_err_ = data[index][2][1]
        #print(len(flux_), len(flux_err_))
        max_index = np.argmax(flux_)
        for j, _ in enumerate(flux_):
            if flux_[j] > 5*flux_err_[j]:
                trigger = mjd_[j]
                break
        #print(trigger, max_index)
        trigger_list.append(trigger)
    return (trigger_list)

trigger_list = trigger(data=shuffle_data_list)
for j, i in enumerate(trigger_list):
    if i>=60:
        fit_fail_list.append(j)

index_list = list(range(len(shuffle_data_list)))
selected_index = []
for index in index_list:
    if index not in fit_fail_list:
        selected_index.append(index)

selected_data_list = []
selected_label_list = []
selected_redshift_list = []
selected_trigger_list = []
selected_interpolate_data_list = []
for i in selected_index:
    selected_interpolate_data_list.append(interpolated_data_list[i])
    selected_trigger_list.append(trigger_list[i])
    selected_data_list.append(shuffle_data_list[i])
    selected_label_list.append(shuffle_label_list[i])
    selected_redshift_list.append(shuffle_redshift_list[i])

selected_weight_list = []
for i in selected_index:
    selected_weight_list.append(weight_list[i])
len(selected_weight_list)

#剪裁/用0填充数据
def generat_lc(sample_size, num_band=3, num_time_steps=int(90)):
    train_data = np.zeros([sample_size, num_time_steps, num_band * 2])
    with tqdm(total=num*10, desc='Processing...', leave=True, ncols=100, unit='B', unit_scale=True) as pbar:
        for index in range(num):
            single_lc = selected_interpolate_data_list[index]
            try:
                for k in range(num_band):
                    for l in range(num_time_steps):
                        if selected_trigger_list[index]>3:
                            train_data[index, l, k] = single_lc[1][k][l+int(selected_trigger_list[index])-3]
                            train_data[index, l, k + num_band] = single_lc[2][k][l+int(selected_trigger_list[index])-3]
                        else:
                            train_data[index, l, k] = single_lc[1][k][l+int(selected_trigger_list[index])]
                            train_data[index, l, k + num_band] = single_lc[2][k][l+int(selected_trigger_list[index])]
            except:
                # 删除第一个轴（样本轴）上的第 index 个元素
                new_arr = np.delete(train_data, index, axis=0)
            pbar.update(10)
    return train_data
print("begain sub\padding data...")
selected_data_array = generat_lc(sample_size=len(selected_data_list))
print('sub\padding data success!')

print("saving data...")
np.savez(f'/home/yltang/data/WFST_lc_classification/data/{train_test}_data.npz', data = selected_data_array)
np.savez(f'/home/yltang/data/WFST_lc_classification/data/{train_test}_weight.npz', data = selected_weight_array)
print("saving success!")

