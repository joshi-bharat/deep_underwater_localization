import os
import cv2

#For Training

# file = '/home/bjoshi/deepcl-data/cyclegan_synth/cyclegan_train.txt'
#
# lines = open(file, 'r').readlines()
# index = 0
#
# train_file = open('/home/bjoshi/deepcl-data/cyclegan_synth/yolo_tf_train.txt', 'w')
#
# root_dir = '/home/bjoshi/deepcl-data/cyclegan_synth/images'
# h = 600
# w = 800
#
# for line in lines:
#     line = line.strip()
#     line = line.split(',')
#     name = line[0]
#     full_path = os.path.join(root_dir, name+'.png')
#     x1 = int(float(line[8])/2)
#     y1 = int(float(line[9])/2)
#     x2 = x1  + int(float(line[10])/2)
#     y2 = y1  + int(float(line[11])/2)
#
#     if x2 > w:
#         continue
#     if x1 < 0:
#         continue
#     if y2 > h:
#         continue
#     if y1 < 0:
#         continue
#     # img = cv2.imread(full_path)
#     # img = cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
#     #
#     # cv2.imshow('Image', img)
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#     train_file.write('%d %s %d %d %d %d %d %d %d\n' % (index, full_path, w, h, 0, x1, y1, x2, y2))
#     if index > 37000:
#         break
#     index += 1
# train_file.close()


#For Validation
file = '/home/bjoshi/deepcl-data/cyclegan_synth/yolo_tf_train.txt'
lines = open(file, 'r').readlines()

label_path = '/home/bjoshi/deepcl-data/cyclegan_synth/labels'
opfile = open('/home/bjoshi/deepcl-data/cyclegan_synth/combined_tf_train.txt','r')
final = open('/home/bjoshi/deepcl-data/cyclegan_synth/final_tf_train.txt','w')
lines = opfile.readlines()
count = 0
for line in lines:
    line = line.strip()
    line_vals = line.split(' ')
    if len(line_vals) < 27:
        count += 1
    else:
        final.write(line)
        final.write('\n')
final.close()
print(count)
#
# height = 600
# width = 800
# for line in lines:
#     line = line.strip()
#     linex = line
#     opfile.write(line)
#     vals = line.split(' ')
#     file_name = vals[1]
#     file_name = os.path.split(file_name)[1].split('.')[0]
#     label = open(os.path.join(label_path, file_name + '.txt' )).readline()
#     label = label.split(' ')
#     for i in range(1, 19):
#         if i % 2 == 0:
#             # print(label[i])
#
#             y = float(label[i]) * height
#             if y < 0 or y > height:
#                 continue
#             opfile.write(' %d' % round(y))
#         else:
#             # print(label[i])
#             x = float(label[i]) * width
#             if x < 0 or x > width:
#                 continue
#             opfile.write(' %d' % round(x))
#
#             # print(x)
#
#     opfile.write('\n')
# opfile.close()