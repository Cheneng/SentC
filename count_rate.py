


TAG_DICT = {'ADJ': 0, 'ADP': 1, 'ADV': 2, 'CONJ': 3, 'DET': 4, 'NOUN': 5,
            'NUM': 6, 'PRT': 7, 'PRON': 8, 'VERB': 9, '.': 10, 'X': 11}
path = './data/tag_save.txt'

count_all_dict = {index: 0 for index in range(12)}
print(count_all_dict)
count_remain_dict = {index: 0 for index in range(12)}
print(count_remain_dict)

with open(path, 'r') as f:
    for index, line in enumerate(f):
        tags, labels = line.split('\t')
        print(index)
        for tag, label in zip(tags.strip().split(' '), labels.strip().split(' ')):
            tag = int(tag)
            count_all_dict[tag] += 1
            if label == '1':
                count_remain_dict[tag] += 1

print(count_all_dict)
print(count_remain_dict)

# with open('./data/all_count.txt', 'w') as f:
#     for item in count_all_dict.items():
#         f.write("{0} ".format(item[1]))
#
# with open('./data/save_count.txt', 'w') as f:
#     for item in count_remain_dict.items():
#         f.write("{0} ".format(item[1]))

{0: 385884, 1: 639117, 2: 105433, 3: 97739, 4: 467565, 5: 2063777, 6: 141125, 7: 166753, 8: 146971, 9: 789354, 10: 513579, 11: 1116}
{0: 101295, 1: 155692, 2: 21640, 3: 18583, 4: 168475, 5: 790652, 6: 42844, 7: 58806, 8: 42096, 9: 419194, 10: 227876, 11: 310}