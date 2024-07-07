import os
import shutil
import time
import csv
import re
# ACDC Dataset
from smtplib import SMTP_SSL
from email.mime.text import MIMEText
import os
import shutil
import time
def sendMail(message,Subject,sender_show,recipient_show,to_addrs,cc_show=''):
    '''
    :param message: str 邮件内容
    :param Subject: str 邮件主题描述
    :param sender_show: str 发件人显示，不起实际作用如："xxx"
    :param recipient_show: str 收件人显示，不起实际作用 多个收件人用','隔开如："xxx,xxxx"
    :param to_addrs: str 实际收件人
    :param cc_show: str 抄送人显示，不起实际作用，多个抄送人用','隔开如："xxx,xxxx"
    '''
    # 填写真实的发邮件服务器用户名、密码
    user = '2226427557@qq.com'
    password = 'smvebzijafqidiaf'
    # 邮件内容
    msg = MIMEText(message, 'plain', _charset="utf-8")
    # 邮件主题描述
    msg["Subject"] = Subject
    # 发件人显示，不起实际作用
    msg["from"] = sender_show
    # 收件人显示，不起实际作用
    msg["to"] = recipient_show
    # 抄送人显示，不起实际作用
    msg["Cc"] = cc_show
    with SMTP_SSL(host="smtp.qq.com",port=465) as smtp:
        # 登录发邮件服务器
        smtp.login(user = user, password = password)
        # 实际发送、接收邮件配置
        smtp.sendmail(from_addr = user, to_addrs=to_addrs.split(','), msg=msg.as_string())


def modeldone():
    # message = text + 'training is done !!!!'
    message = '10num training is done !!!!'
    Subject = 'hello'
    # 显示发送人
    sender_show = '2226427557@qq.com'
    # 显示收件人
    recipient_show = '2226427557@qq.com,844505032@qq.com'
    # 实际发给的收件人
    to_addrs = '2226427557@qq.com,844505032@qq.com'
    sendMail(message,Subject,sender_show,recipient_show,to_addrs)

def is_folder_empty(folder_path):
    return not any(os.listdir(folder_path))

def extract_accuracy_from_line(line):
    match = re.search(r'accuracy/top1: (\d+\.\d+)', line)
    return float(match.group(1)) if match else None

def find_highest_accuracy_epoch(lines):
    max_accuracy = 0.0
    max_accuracy_epoch = None

    for line in lines:
        accuracy = extract_accuracy_from_line(line)
        if accuracy is not None and accuracy > max_accuracy:
            max_accuracy = accuracy
            epoch_match = re.search(r'test_epoch(\d+)\.log', line)
            max_accuracy_epoch = epoch_match.group(1) if epoch_match else None

    return max_accuracy

def find_corresponding_pkl(folder, max_accuracy_epoch):
    pkl_files = [file for file in os.listdir(folder) if file.startswith(f'out_epoch{max_accuracy_epoch}') and file.endswith('.pkl')]

    if not pkl_files:
        print(f"文件夹 '{folder}' 中找不到对应最高accuracy的pkl文件.")
        return None

    return os.path.join(folder, pkl_files[0])

def process_folders(folder_list):
    corresponding_pkl_files = []
    folder_list = [folder_list]
    for folder in folder_list:
        files = os.listdir(folder)
        csv_files = [file for file in files if file.endswith('alltest.csv')]

        if not csv_files:
            print(f"文件夹 '{folder}' 中找不到以'alltest.csv'结尾的文件.")
            continue

        csv_file = os.path.join(folder, csv_files[0])

        with open(csv_file, 'r') as file:
            lines = file.readlines()

        max_accuracy = find_highest_accuracy_epoch(lines)

    return max_accuracy

def runtrainandtest(ptname, configfile, workpath):
    os.makedirs(workpath, exist_ok=True)
    nowlog = os.path.join(workpath, "train.log")
    os.system(f'nohup ./dist_train.sh {configfile} 2 --work-dir {workpath} >{nowlog} 2>&1 &')
    targetpt = os.path.join(workpath, "epoch_50.pth")
    # print(targetpt)
    while not os.path.exists(targetpt):
    # print(f"等待目标文件 {targetpt} 生成...")
        time.sleep(30)  # 等待5分钟，单位为秒
    time.sleep(60)  # 等待5分钟，单位为秒
    #     time.sleep(1)  # 等待5分钟，单位为秒
    # time.sleep(1)  # 等待5分钟，单位为秒
        # print(f"{targetpt} 已生成，继续执行后续操作。")
        # time.sleep(50) 
    ptpath = workpath
    for nowpt in ptname:
        nownum = nowpt.split("_")[1].split(".")[0]
        nowpt = os.path.join(ptpath, nowpt)
        nowpkl = os.path.join(ptpath, "out_epoch" + nownum + ".pkl")
        nowlog = os.path.join(ptpath, "test_epoch" + nownum + ".log")
        os.system(f'nohup ./dist_test.sh {configfile} {nowpt} 2 --out {nowpkl} >{nowlog} 2>&1 &')
        targetpt = nowpkl
        while not os.path.exists(targetpt):
            # print(f"等待目标文件 {targetpt} 生成...")
            time.sleep(3)  # 等待5分钟，单位为秒
        # print(f"目标文件 {targetpt} 已生成...")
        time.sleep(15)  # 等待5分钟，单位为秒
        #     time.sleep(1)  # 等待5分钟，单位为秒
        # # print(f"目标文件 {targetpt} 已生成...")
        # time.sleep(1)  # 等待5分钟，单位为秒
    nowpath = workpath
    name = nowpath.split("/")[-1]
    dir_list = os.listdir(nowpath)
    nowcsv = {}
    for cur_file in dir_list:
        newpath = os.path.join(nowpath, cur_file)
        if os.path.isfile(newpath) :
            if cur_file[:10] == "test_epoch":
                nowcount = int(cur_file.split(".")[0].split("test_epoch")[1])
                alltsr = []
                # Open file 
                fileHandler  =  open  (newpath, "r", encoding="utf-8")
                while  True:
                    # Get next line from file
                    line  =  fileHandler.readline()
                    # If line is empty then end of file reached
                    if  not  line  :
                        break;
                    # print(line.strip())
                    alltsr.append(line.strip())
                    # Close Close    
                fileHandler.close()
                # print(len(alltsr))
                # alltsr = alltsr[-10]
                alltsr = alltsr[-1]
                nowcsv[nowcount] =  [cur_file, alltsr]
            else:
                continue  
    sorted(nowcsv.keys())
    writecsv = list(nowcsv.values())
    with open(os.path.join(nowpath,name + "_" + "alltest.csv"), "w", encoding="utf-8", newline="") as f:
        csv_writer = csv.writer(f)
        writecsv.append(name)
        csv_writer.writerows(writecsv)
    f.close()
    max_accuracy = process_folders(workpath)
    # print(max_accuracy)
    return max_accuracy

ptname = [
          "epoch_5.pth","epoch_10.pth","epoch_15.pth","epoch_20.pth","epoch_25.pth",
          "epoch_30.pth","epoch_35.pth","epoch_40.pth", "epoch_45.pth","epoch_50.pth"]
ACDCconfigfile_list = [
               r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py",
            #    r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py",
            #    r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py",
            #    r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py",
            #    r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py",
            #    r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py",
                ]
ACDCworkpath_list_all = [
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A1/A1_01",
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A1/A1_02",
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A1/A1_06",

            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A2/A2_025",
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A2/A2_2",
            
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A3/A3_1",
            r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A3/A3_40",
]
targetpth_list = [
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A1/A1_01/A1_01_alltest.csv",
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A1/A1_02/A1_02_alltest.csv",
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A1/A1_06/A1_06_alltest.csv",

            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A2/A2_025/A2_025_alltest.csv",
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A2/A2_2/A2_2_alltest.csv",
            
            # r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A3/A3_1/A3_1_alltest.csv",
            r"/root/lzz2/SKGC_new/weight/classification/ablation_A/A3/A3_40/A3_40_alltest.csv",
               ]
demopy = "image_demo_with_inferencer.py"
demopath = r"/root/lzz1/Diseases_lzz/mmsegmentation/demo"
pypath_list = [
            # r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A1/A1_01",
            # r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A1/A1_02",
            # r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A1/A1_06",

            # r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A2/A2_025",
            # r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A2/A2_2",
            
            # r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A3/A3_1",
            r"/root/lzz2/SKGC_new/weight/segmentation/ablation_A/A3/A3_40",
]
# threlist = [98, 97.5, 98, 98, 97.5, 97]
threlist = [97.4]
for nowconfig,nowworkpath,nowtarget,nowdemo,nowthre in zip(ACDCconfigfile_list,ACDCworkpath_list_all,targetpth_list, pypath_list, threlist):
    nowdemopy = os.path.join(nowdemo,demopy)
    tgdemopy = os.path.join(demopath,demopy)
    shutil.copy(nowdemopy,tgdemopy)
    os.makedirs(nowworkpath, exist_ok=True)
    while is_folder_empty(nowworkpath):
        print("runing   !  !")
        max_accuracy = runtrainandtest(ptname, nowconfig, nowworkpath)
        while not os.path.exists(nowtarget):
            # print(f"等待目标文件 {targetpt} 生成...")
            time.sleep(20)
        print(f"The maximum Accuracy in the CSV file is: {max_accuracy}")
        with open('run_trainW.csv', 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([max_accuracy]) 
        if float(max_accuracy) > nowthre:
            # print(max_accuracy)
            os.system(f'rm -rf {nowworkpath}/*')
modeldone()

configfile1 = r"/root/lzz1/Diseases_lzz/mmpretrain/configs/eva02/eva02-tiny-p14_in1k.py"
workpath1 = r"/root/lzz2/SKGC_new/weight/classification/test"
nowlog1 = r"/root/lzz2/SKGC_new/weight/classification/test/train.log"
while True:
    os.system(f'./dist_train.sh {configfile1} 2 --work-dir {workpath1}')