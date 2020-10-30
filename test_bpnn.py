#@Time:10/29/20205:37 PM
#@Author: Mini(Wang Han)
#@Site:
#@File:test_forth_nn.py
input_notes=[0.8,1.0,0.4]
weight_0=[[0.1,0.2,0.3],[-0.2,-0.1,0.1],[0.1,-0.1,0.2,]]
weight_1=[0.3,0.5,0.4]

#计算隐藏层节点输入
def hide_layer_input(input_notes,weight_0):
    hide_note=[]
    for j in range(0, len(weight_0)):
        note = 0
        k = 0
        for i in range(0, len(input_notes)):
            note = note + input_notes[i] * weight_0[j][k]
            k = k + 1

        hide_note.append(note)
    print("隐藏层输入节点：", hide_note)
    return hide_note

import math
#激活函数
def sig(note):
    out_sig=1.0 / (1.0 + math.exp(-note))
    return out_sig
#计算隐藏层输出节点、输出层输入&输出节点
def output_layer(hide_note,weight_1):
    output_node = 0
    input_hide = []
    m = 0
    for i in hide_note:
        output_node = output_node + sig(i) * weight_1[m]
        m = m + 1
        input_hide.append(sig(i))
    output_node_ = sig(output_node)
    print("隐藏层输出节点：", input_hide)
    print("输出层输入节点：", output_node)
    print("输出层输出节点：", output_node_)
    return input_hide,output_node,output_node_

#sigmod 求导
def sigmod_derivate(x):
    return x * (1 - x)
#将float,转为可以discribe的列
def round_list(list1,n=4):
    lista = []
    for i in list1:
        lista.append(round(i,n))
    return lista
#计算输出层和隐层的误差error
def error_output(y,output_calculate):
    error=(y-output_calculate)*sigmod_derivate(output_calculate)
    print("输出层误差:",error)
    return error
def error_hidden(output_error,hiden_output,weight_1):
    error=[]
    k=0
    weight_1=round_list(weight_1)
    for i in hiden_output:
        error_=output_error*weight_1[k]*sigmod_derivate(i)
        error.append(error_)
        k=k+1
    print("隐藏层误差：",error)
    return error

#计算Delta Rule下的矫正误差(r为学习率)
def change_error(r,error_output_,output_note,weight_orignal):
    error=[]
    weight=[]
    output_note=round_list(output_note)

    for i in range(0,len(output_note)):
        change_error_=r*error_output_*output_note[i]
        error.append(change_error_)
        weight.append(weight_orignal[i]+change_error_)
    #print("weight的矫正值为：",error)
    #print("weight矫正后：", weight)
    return weight

if __name__ == '__main__':
    hide_note=hide_layer_input(input_notes,weight_0)
    hiden_output, output_node, output_node_=output_layer(hide_note, weight_1)
    error_output_=error_output(y=0.67,output_calculate=output_node_)
    #隐藏层误差值计算
    error_hidden_=error_hidden(error_output_,hiden_output,weight_1)
    #weight_1的矫正值
    weight_1_weight=change_error(0.3,error_output_,hiden_output,round_list(weight_1))#学习率设置为0.3
    print("weight_1校正后的值：",weight_1_weight)
    #weight_0的矫正值
    weight_0_weight=[]
    for i in range(0, len(weight_0)):
        weight = change_error(0.3, error_hidden_[i], input_notes, weight_orignal=round_list(weight_0[i]))
        print(weight_0[i], error_hidden_[i])
        weight_0_weight.append(weight)
    print("weight_0校正后的值：",weight_0_weight)





