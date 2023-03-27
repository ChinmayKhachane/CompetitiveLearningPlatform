import pycode_similar
import os
import time

#prints out percentages of plagiarized code to resulting textfile

def check_plagiarism(files, ressy, filenames):
    sum = pycode_similar.detect(files, diff_method=pycode_similar.UnifiedDiff, keep_prints=False, module_level=False)
    result = open(ressy, 'a')
    result.write('\n ref : {} \n\n'.format(filenames[0]))
    for index, func_ast_diff_list in sum:
        sum_plagiarism_percent, sum_plagiarism_count, sum_total_count = pycode_similar.summarize(func_ast_diff_list)
        result.write('candidate : ' + filenames[index] + '\n')
        result.write('{:.2f} % ({}/{}) of ref code structure is plagiarized by candidate.\n'.format(
            sum_plagiarism_percent * 100,
            sum_plagiarism_count,
            sum_total_count,
        ))


 # returns list with actual code from file and list with all filenames   

def get_Files():
    path = "C:\\Users\\Chinmay\\OneDrive\\Desktop\\testcases"
    dir_list = os.listdir(path)
    newl = [x for x in dir_list if x.endswith('.py')]
    the_f = []
    for i in range(len(newl)):
        x = open(newl[i], 'r').read()
        the_f.append(x)
    return [the_f, newl]


filestoput, indexes = get_Files()[0], get_Files()[1]


resi = 'temp.txt'

start = time.time()
for i in range(len(indexes)):
    check_plagiarism(filestoput, resi, indexes)
    filestoput.append(filestoput.pop(0))
    indexes.append(indexes.pop(0))
end = time.time()

print(end-start)
