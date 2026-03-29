import csv
def save2csv(results, output_path):
    with open(output_path, mode='w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        # 建议增加类别和坐标列，方便后期排查哪个人脸曝光不对
        writer.writerow(["image_name", "class_id", "roi_box", "brightness", "status"])
        writer.writerows(results) # 语法点：writerows 可以一次性写入整个列表，效率更高