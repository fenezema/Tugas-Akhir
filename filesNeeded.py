import os

txt = open('filesNeeded.txt','w')

files = os.listdir('.')

for element in files:
    txt.write(element)
    if "cache" in element:
        break
    txt.write('\n')
    

txt.close()
print("filesNeeded.txt has been made")