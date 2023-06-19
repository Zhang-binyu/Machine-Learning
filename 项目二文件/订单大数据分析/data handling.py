import json

import numpy as np
import pandas as pd
import xlwt

path_OrderModel = r"OrderModel订单表.csv"
path_CommodityModel = r"CommodityModel商品表.csv"


def count_head(CustomName):
    # 用来统计店铺类别
    temp = []
    for i in range(len(CustomName)):
        if CustomName[i] not in temp:
            temp.append(CustomName[i])
    print("一共" + str(len(temp)) + "家店铺")
    return temp, len(temp)


def goods_name(data_list):
    temp = []
    for i in range(len(data_list)):
        if data_list[i] not in temp:
            temp.append(data_list[i])
    return temp, len(temp)


def main():
    data = pd.read_csv(path_OrderModel)
    CustomName = data.pop("CustomName")
    CreateTime = data.pop("CreateTime")
    Goods = data.pop("Goods")

    data_CommodityModel = pd.read_csv(path_CommodityModel)
    Goods_Name = np.array(data_CommodityModel.pop('Name'))

    # 统计所有店名
    List_Name = count_head(CustomName)

    all_goods = []
    for i in range(len(Goods)):
        # if CustomName[i] == "青羊区周记蜀大侠火锅店(西玉龙二店直营店）":
        #     pass
        Goods_test = Goods[i]
        Goods_list = json.loads(Goods_test)

        CustomName_one = CustomName[i]
        CreateTime_one = (str(CreateTime[i][0:-5])).strip()

        for j in range(len(Goods_list)):
            all_goods.append([CreateTime_one, CustomName_one, Goods_list[j]['GoodsName'], Goods_list[j]['Qty']])

    # 将查询结果写入到excel
    workbook = xlwt.Workbook(encoding='GBK')
    # 创建一个新的sheet
    sheet = workbook.add_sheet('sheet1', cell_overwrite_ok=True)
    # 写上 标头
    sheet.write(0, 0, "时间")
    sheet.write(0, 1, "店名")
    sheet.write(0, 2, "产品")
    sheet.write(0, 3, "个数")

    # 统计数目
    # for i in range(len(Goods_Name)):
    #     num = all_goods.count(Goods_Name[i])
    #     sheet.write(i+1, 0, Goods_Name[i])
    #     sheet.write(i+1, 1, num)

    for i in range(len(all_goods)):
        sheet.write(i + 1, 0, all_goods[i][0])
        sheet.write(i + 1, 1, all_goods[i][1])
        sheet.write(i + 1, 2, all_goods[i][2])
        sheet.write(i + 1, 3, all_goods[i][3])

    # excel保存为文件
    workbook.save(r'D:\mycode\untitled7\商品货单表.xls')


if __name__ == "__main__":
    main()
