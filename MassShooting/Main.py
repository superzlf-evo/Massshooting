import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import pickle
import os
import plotly.express as px
# from sklearn.inspection import permutation_importance
# from eli5.sklearn import PermutationImportance
import shap
from keras.models import load_model
def Graph_Of_Income():
    print("请选择选项：")
    print("1.热力图")
    print("2.折线图")
    Chose = input("输入选项：")
    if Chose=='1':
        df = pd.read_excel('Income/income.xlsx')
        fig = px.choropleth(df,
                            locations="Code",
                            color="income",
                            animation_frame="Year",
                            color_continuous_scale="ylorbr",
                            locationmode="USA-states",
                            scope="usa",
                            range_color=(20000, 80000),
                            title="Per Capita Personal Income by States and Years",
                            height=600)

        fig.show()
        print("图已展示完毕")
        # fig.write_html("PCI_Choropleth.html")
    elif Chose=='2':
        df_income = pd.read_excel('Income/income_year.xlsx')
        df_shooting = pd.read_excel('Income/shooting_year.xlsx')
        year = df_income['year'].tolist()
        income = df_income['income'].tolist()
        number = df_shooting['number'].tolist()

        fig = plt.figure(figsize=(8, 6))

        ax1 = fig.add_subplot(111)
        ax1.plot(year, income, marker='o', color='b', label='income')
        ax1.set_ylim([40000, 70000])
        ax1.set_ylabel('income')
        ax1.set_title("Trend Of Per Capita Personal Income And Mass Shooting Incident Number By Years")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(year, number, marker='*', color='r', label='number of shootings')
        ax2.set_ylim([200, 1000])
        ax2.set_ylabel('number of shootings')
        ax2.set_xlabel('year')
        ax2.legend(loc='upper right')

        plt.show()
        # plt.savefig('Trend Of Per Capita Personal Income Compared to MSI(2014-2022).png', dpi=300, format=('png'))
        print("图已展示完毕")
    else:
        print("输入有误，请重新操作")


def Graph_Of_Population():
    print("请选择选项：")
    print("1.热力图")
    print("2.折线图")
    Chose = input("输入选项：")
    if Chose == '1':
        dfp = pd.read_csv('Population/Population.csv', encoding='Latin1')
        fig = px.choropleth(dfp,
                            locations="Code",
                            color="Population",
                            animation_frame="Year",
                            color_continuous_scale="pubu",
                            locationmode="USA-states",
                            scope="usa",
                            range_color=(500000, 11000000),
                            title="Population by States and Years",
                            height=600)
        fig.show()
        # fig.write_html('Population_Choropleth(2014-2022).html')

        print("图已展示完毕")
    elif Chose == '2':
        dfp = pd.read_csv('Population/Population.csv', encoding='Latin1')
        # dfc = pd.read_csv('Population/Crime_counter.csv', encoding='Latin1')


        dfl = dfp.groupby('Year', as_index=False).agg({'Population': 'sum'})
        df_shooting = pd.read_excel('Population/shooting_year.xlsx')
        year = dfl['Year'].tolist()
        population = dfl['Population'].tolist()
        number = df_shooting['number'].tolist()

        fig = plt.figure(figsize=(8, 6))

        ax1 = fig.add_subplot(111)
        ax1.plot(year, population, marker='o', color='b', label='population')
        ax1.set_ylim([310000000, 340000000])
        ax1.set_ylabel('population')
        ax1.set_title("Trend Of Population And Mass Shooting Incident Number By Years")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(year, number, marker='*', color='r', label='number of shootings')
        ax2.set_ylim([200, 1000])
        ax2.set_ylabel('number of shootings')
        ax2.set_xlabel('year')
        ax2.legend(loc='upper right')

        plt.show()
        print("图已展示完毕")
    else:
        print("输入有误，请重新操作")


def Graph_Of_EM_Rate():
    print("请选择选项：")
    print("1.热力图")
    print("2.折线图")
    Chose = input("输入选项：")
    if Chose == '1':
        df1 = pd.read_csv('EM_Rate/new.csv', encoding='Latin1')
        df2 = pd.read_excel('EM_Rate/income.xlsx')
        df_code = df2[['Code', 'State']]
        df = pd.merge(df1, df_code)
        df = df.drop_duplicates()

        fig = px.choropleth(df,
                            locations="Code",
                            color="Rate",
                            animation_frame="Year",
                            color_continuous_scale="turbo",
                            locationmode="USA-states",
                            scope="usa",
                            range_color=(0, 10),
                            title="Unemployment Rate by States and Years",
                            height=600)

        fig.show()
        # fig.write_html('UNEMRATE_Choropleth(2014-2023).html')
        print("图已展示完毕")
    elif Chose == '2':
        df_in = pd.read_csv('EM_Rate/us_unemployment.csv')
        # 14-23年每年失业率数据交互图
        df2 = df_in.copy()
        # 将time列转换为datetime类型
        df2['time'] = pd.to_datetime(df2['time'])
        # 提取年份信息并创建一个新的列year
        df2['year'] = df2['time'].dt.year
        # 使用groupby方法按年份分组并计算每年的失业率平均值
        result = df2.groupby('year')['rate'].mean().reset_index()
        fig2 = px.line(result, x='year', y='rate', title='Unemployment Rate of USA')
        # fig2.show()
        # fig2.write_html('UNEMRATE_Linechart(2014-2023).html')

        dfl = pd.read_csv('EM_Rate/Year_data.csv', encoding='Latin10')
        dfl = dfl.groupby('Year', as_index=False).agg({'Rate': 'mean'})
        dfl = dfl[dfl['Year'] != 2023]
        df_shooting = pd.read_excel('Income/shooting_year.xlsx')
        year = dfl['Year'].tolist()
        rate = dfl['Rate'].tolist()
        number = df_shooting['number'].tolist()

        fig = plt.figure(figsize=(8, 6))

        ax1 = fig.add_subplot(111)
        ax1.plot(year, rate, marker='o', color='b', label='rate')
        ax1.set_ylim([0, 10])
        ax1.set_ylabel('rate/%')
        ax1.set_title("Trend Of Unemployment Rate And Mass Shooting Incident Number By Years")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(year, number, marker='*', color='r', label='number of shootings')
        ax2.set_ylim([200, 1000])
        ax2.set_ylabel('number of shootings')
        ax2.set_xlabel('year')
        ax2.legend(loc='upper right')

        plt.show()
        print("图已展示完毕")
        # plt.savefig('Trend Of Unemployment Rate Compared to MSI(2014-2022).png', dpi=300, format=('png'))
    else:
        print("输入有误，请重新操作")

def Graph_Of_GDP():
    print("请选择选项：")
    print("1.热力图")
    print("2.折线图")
    Chose = input("输入选项：")
    if Chose == '1':

        df = pd.read_excel('GDP/GDP 2014-2022.xlsx')
        fig = px.choropleth(df,
                            locations="code",
                            color="GDP",
                            animation_frame="year",
                            color_continuous_scale="hot_r",
                            locationmode="USA-states",
                            scope="usa",
                            range_color=(0, 15000000),
                            title="US GDP by States and Years",
                            height=600)
        fig.show()
        print("图已展示完毕")

    elif Chose == '2':
        dfl = pd.read_excel('GDP/ALL GDP_year.xlsx')
        df_shooting = pd.read_excel('Income/shooting_year.xlsx')
        year = dfl['YEAR'].tolist()
        GDP = dfl['GDP'].tolist()
        number = df_shooting['number'].tolist()

        fig = plt.figure(figsize=(8, 6))

        ax1 = fig.add_subplot(111)
        ax1.plot(year, GDP, marker='o', color='b', label='GDP')
        ax1.set_ylim([65000000, 82500000])
        ax1.set_ylabel('population')
        ax1.set_title("Trend Of GDP And Mass Shooting Incident Number By Years")
        ax1.legend(loc='upper left')

        ax2 = ax1.twinx()  # this is the important function
        ax2.plot(year, number, marker='*', color='r', label='number of shootings')
        ax2.set_ylim([200, 1000])
        ax2.set_ylabel('number of shootings')
        ax2.set_xlabel('year')
        ax2.legend(loc='upper right')

        plt.show()
        # plt.savefig('Trend Of GDP Compared to MSI(2014-2022).png', dpi=300, format=('png'))
        print("图已展示完毕")

    else:
        print("输入有误，请重新操作")


def Graph_Of_Count():
    print("请选择选项：")
    print("1.热力图")
    print("2.柱状图")
    Chose = input("输入选项：")
    if Chose == '1':

        dfc = pd.read_csv('Population/Crime_counter.csv', encoding='Latin1')
        fig = px.choropleth(dfc,
                            locations="Code",
                            color="Num",
                            animation_frame="Year",
                            color_continuous_scale="reds",
                            locationmode="USA-states",
                            scope="usa",
                            range_color=(0, 100),
                            title="Mass shooting incident numbers by States and Years",
                            height=600)
        fig.show()
        # fig.write_html('MSIN_Choropleth(2014-2022).html')
        print("图已展示完毕")
    elif Chose == '2':
        # 读取数据
        data = pd.read_excel('Income/shooting_year.xlsx')

        # 创建画布和子图
        fig, ax = plt.subplots(figsize=(8, 5))

        # 绘制柱状图
        rects = ax.bar(x=data['year'], height=data['number'], color='blue')

        # 设置图表标题和坐标轴标签
        ax.set_title('Total Gun Violence Incidents in the US (2014-2022)')
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of incidents')

        # 显示数据标签
        for rect in rects:
            height = rect.get_height()
        ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
        xytext = (0, 3), textcoords = "offset points", ha = 'center', va = 'bottom')

        # 显示图表
        plt.show()
        print("图已展示完毕")
    else:
        print("输入有误，请重新操作")

def Graph_Of_2023Count():
    print("请选择选项：")
    print("1.气泡图展示2023年预测结果")
    print("2.2023年预测结果与2023年真实结果前三季度对比")
    Chose = input("输入选项：")
    if Chose == '1':
        df4 = pd.read_csv('Data2023test.csv')
        df2 = pd.read_csv('Count2023_data.csv')
        df3 = pd.read_excel('Latitude_Longitude.xlsx')
        df1 = df4[['Year', 'Quarter', 'State']]
        df1['Count'] = df2['Count']
        # print(df1)

        df = pd.merge(df1, df3)
        # print(df)
        for i in range(int(len(df) / 4)):
            sum = 0
            sum += df.iloc[i * 4]['Count']
            sum += df.iloc[i * 4 + 1]['Count']
            sum += df.iloc[i * 4 + 2]['Count']
            sum += df.iloc[i * 4 + 3]['Count']
            # print(sum)

            df.iloc[i * 4, 3] = sum
        # df['Count new'] = df['Count']
        # print(df5)
        df = df[df['Quarter'] <= 1]
        # print(df)
        fig = px.scatter_geo(df, lon='Longitude', lat='Latitude', color='Count', range_color=[0, 50],
                             hover_name='State', size='Count', size_max=40, animation_frame='Year',
                             locationmode='USA-states', scope='usa')
        fig.show()
        print("图已展示完毕")
        # fig.write_html("data/Data2023test03.html")
    elif Chose == '2':
        count_result = pd.read_csv('Count2023_data.csv', encoding='Latin1')
        counter = 1
        for num in count_result['Count']:
            counter += 1
            if counter % 4 == 0:
                count_result = count_result.drop(index=counter - 1)
        count_result = count_result.reset_index()
        count_fact = pd.read_csv('CountQ1-3Real.csv', encoding='Latin1')
        df = pd.DataFrame(count_fact['Count'])
        df['Count-Result'] = count_result['Count']

        cf = df['Count'].tolist()
        cr = df['Count-Result'].tolist()

        plt.figure(figsize=(20, 5))
        plt.plot(cf, marker='o', linestyle='-', color='b', alpha=0.6, label='Fact')
        plt.plot(cr, marker='o', linestyle='--', color='r', alpha=0.6, label='Result')
        plt.title('Comparison between Result and Fact (2023 1-3 Quarter)')
        plt.ylabel('Count number')
        plt.xticks([])
        plt.legend(loc='upper left')
        plt.show()
        # plt.savefig('Comparison between Result and Fact(2023 1-3 Quarter).png', dpi=300, format=('png'))
        print("图已展示完毕")
    else:
        print("输入有误，请重新操作")






def TranMyModel():
    # 加载数据
    data = pd.read_csv('MassShootingall.csv')
    data2023 = pd.read_csv('Data2023test.csv')
    # print(data.info())

    for i in range(len(data)):
        data.iloc[i, 4] = int(int(data.iloc[i, 4].replace(',', '')) / 10000)#人口单位变为万
        data.iloc[i, 5] = int(float(data.iloc[i, 5]) *100)#失业率*100
        data.iloc[i, 6] = int((data.iloc[i, 6]) / 1000)
        data.iloc[i, 7] = int((data.iloc[i, 7]) / 1000)

    for i in range(len(data2023)):
        data2023.iloc[i, 1] = int(data2023.iloc[i, 1] + 35)
        data2023.iloc[i, 3] = int(int(data2023.iloc[i, 3].replace(',', '')) / 10000)#人口单位变为万
        data2023.iloc[i, 4] = int(float(data2023.iloc[i, 4]) *100)#失业率*100
        data2023.iloc[i, 5] = int((data2023.iloc[i, 5]) /1000)
        data2023.iloc[i, 6] = int((data2023.iloc[i, 6]) / 1000)

    # 将年份和季度合并为一个新的特征
    data['Year_Quarter'] = data['Year'].astype(str) + '-' + data['Quarter'].astype(str)

    # 将特征转换为数字编码
    State_map = {State: index for index, State in enumerate(data['State'].unique())}
    data['State_encoded'] = data['State'].map(State_map)
    data2023['State_encoded'] = data2023['State'].map(State_map)
    year_quarter_mapping = {year_quarter: i for i, year_quarter in enumerate(data['Year_Quarter'].unique())}
    data['Year_Quarter'] = data['Year_Quarter'].map(year_quarter_mapping)
    # print(data)
    # print(data2023)
    # 指定你的特征变量
    x = data[['Year_Quarter','State_encoded','Population','EM_Rate','Income','GDP']]
    x = np.array(x, dtype=np.int64)
    y = data['Count']
    y = np.array(y, dtype=np.int64)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=50)
    #####################################################################
    #定义模型
    model = Sequential()
    model.add(Dense(64, input_dim=6, activation='PReLU'))  # 输入层
    model.add(Dense(128, activation='PReLU'))  # 隐藏层
    model.add(Dense(1, activation='PReLU'))  # 输出层
    model.compile(loss='mean_squared_error', optimizer='adam')  # 使用均方误差作为损失函数，优化器为Adam
    #########################################
    print("是否使用预训练模型")
    jiazai = input("y/n：")
    if jiazai == 'y':  # 调用已保存的训练好的模型
        if is_file_exists("model.pkl") == 0 or is_file_empty("model.pkl") == 1:
            print("文件为空")
        else:
            # 加载模型
            # model = load_model('model.pkl')
            with open('model.pkl', 'rb') as file:
                model = pickle.load(file)
            print("加载模型完成")

    else:
        ipepochs = input("输入训练次数：")  # 次数
        ipbatch_size = input("输入batch_size：")  # 样本大小
        model.fit(x_train, y_train, epochs=int(ipepochs), batch_size=int(ipbatch_size))  # 开始训练模型
        # loss = model.evaluate(x_test, y_test)
        y_pred = model.predict(x_test)
        # Sum = 0
        for i in range(len(y_pred)):
            y_pred[i] = int(y_pred[i][0])
            if y_pred[i] < 0:
                y_pred[i] = 0
            # Sum += abs(y_pred[i] - y_test[i])
        # mse = mean_squared_error(y_test, y_pred)
        # print('Mean Squared Error:', mse)
        # print(Sum / len(y_pred))
        # for i in range(len(y_pred)):
        #     print(int(y_pred[i][0]), y_test[i])
        # perm = PermutationImportance(model, random_state=42).fit(x_train, y_train)
        # eli5.show_weights(perm, feature_names=['Year_Quarter', 'State_encoded', 'Population'])

        # explainer = shap.KernelExplainer(model.predict, x_train)
        # # 计算SHAP值
        # shap_values = explainer.shap_values(x_test)
        # shap_len = len(shap_values)
        # shap_sum = [0, 0, 0, 0,0]

        # for i in range(shap_len):
        #     shap_sum[0] += shap_values[0][i][0]
        #     shap_sum[1] += shap_values[0][i][1]
        #     shap_sum[2] += shap_values[0][i][2]
        #     shap_sum[3] += shap_values[0][i][3]
        #     shap_sum[4] += shap_values[0][i][4]
        #
        # print(shap_sum[0] / shap_len, shap_sum[1] / shap_len, shap_sum[2] / shap_len, shap_sum[3] / shap_len,shap_sum[4] / shap_len)
        #计算模型训练准确度
        print("模型已经训练完成")
        acsum = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                acsum += 1
        print("Accuracy:", acsum / len(y_pred))
        ac_1sum = 0
        for i in range(len(y_pred)):
            if abs(y_pred[i] - y_test[i]) <= 1:
                ac_1sum += 1
        print("Accuracy误差在1之间:", ac_1sum / len(y_pred))
        ac_2sum = 0
        for i in range(len(y_pred)):
            if abs(y_pred[i] - y_test[i]) <= 2:
                ac_2sum += 1
        print("Accuracy误差在2之间:", ac_2sum / len(y_pred))

        ###################################################################
        print("是否保存模型")
        baocun = input("y/n：")
        if baocun == 'y':
            filename = 'model.pkl'
            if os.path.isfile(filename):
                # 如果文件存在，则删除文件
                os.remove(filename)
            # model.save('model.pkl')
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            print("模型已保存到文件")
        else:
            print("模型未保存")

    ##################################################

    print("是否预测2023年各州各季度的大型枪击事件数？")
    pred2023 = input("y/n：")
    if pred2023 == 'y':
        input2023 = data2023[['Quarter', 'State_encoded','Population','EM_Rate','Income','GDP']]
        input2023 = np.array(input2023, dtype=np.int64)
        Count2023 = model.predict(input2023)
        for i in range(len(Count2023)):
            Count2023[i][0] = int(Count2023[i][0])
            if Count2023[i] < 0:
                Count2023[i] = 0
            # print(input2023[i], Count2023[i][0])

        df1 = pd.read_csv('Data2023test.csv')
        df_2023 = pd.DataFrame(Count2023)
        df_2023.columns = ["Count"]
        df1["Count"]= df_2023["Count"]
        print(df1)
        # df1.to_csv('Count2023alldata.csv', index=False)


        print("是否保存预测值？")
        Tosave2023 = input("y/n：")
        if Tosave2023 == 'y':

            filename = 'Count2023_data.csv'
            if os.path.isfile(filename):
                # 如果文件存在，则删除文件
                os.remove(filename)
            with open(filename, 'wb') as file:
                df_Count2023 = pd.DataFrame(Count2023)
                df_Count2023.columns =["Count"]
                df_Count2023.to_csv('Count2023_data.csv', index=False)
            print("已保存到文件")
        else:
            print("下次一定！！！")

    else:
        print("下次一定！！！")



def is_file_exists(file_path):
    return os.path.exists(file_path)

'''判断文件是否为空'''
def is_file_empty(file_path):
    return os.stat(file_path).st_size == 0


def showOptions():
    print("----------------------")

    print("请选择以下功能：")
    print("1.14-22年美国各州的收入展示图")
    print("2.14-22年美国各州的失业率展示图")
    print("3.14-22年美国各州的人口展示图")
    print("4.14-22年美国各州的GDP展示图")
    print("5.14-22年美国各州的大型枪击事件数目展示图")
    print("6.2023大型枪击事件数量预测模型训练")
    print("7.关于2023美国各州大型枪击事件数量预测图")
    print("8.退出")
    print("----------------------")



###################Main#############################

print("欢迎进入系统")
while True:
    showOptions()
    choice = input('选你所想，为所欲为: ')
    # with conn:
    if choice == '8':
        print("期待再次使用")
        break
    elif choice == '1':
        Graph_Of_Income()
    elif choice == '2':
        Graph_Of_EM_Rate()
    elif choice == '3':
        Graph_Of_Population()
    elif choice == '4':
        Graph_Of_GDP()
    elif choice == '5':
        Graph_Of_Count()
    elif choice == '6':
        TranMyModel()
    elif choice == '7':
        Graph_Of_2023Count()
    else:
        print('输入错误，请重新输入')
        continue
######################################################