import pycode_similar
import os

def check_plagiarism(files, ressy):
    the_f = []
    for i in range(len(files)):
        x = open(files[i], 'r').read()
        the_f.append(x)
    result = open(ressy, 'w')
    sum = pycode_similar.detect(the_f, diff_method=pycode_similar.UnifiedDiff, keep_prints=False, module_level=False)
    d = dict((x,y) for x,y in sum)
    for i in d.values():
        nus = str((i[0]))
        result.write(nus + '\n')   
    result.close()

def get_Files():
    path = "C:\\Users\\Chinmay\\testcases"
    dir_list = os.listdir(path)
    return [x for x in dir_list if x.endswith('.py')]
   

filestoput = get_Files()
filestoput.pop(0)



resi = 'temp.txt'


check_plagiarism(filestoput, resi)





