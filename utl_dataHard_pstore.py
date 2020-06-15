import sys, ast, os
import numpy as np
import locale
from datetime import datetime
#from scipy import stats
# import simplejson as json

locale.setlocale(locale.LC_TIME, 'pt_BR.utf8')

class_file = open('pstore_utl.classes','w')
# pos_file = open('filmow_utl.pos','w')
err_file = open('pstore_utl.err','w')

path = sys.argv[1]
#path = 'play_store'
folders = os.listdir(path)

cont_util = 0
cont_nutil = 0

contador = 0
for folder in folders:
  total = []
  files = []
  paths = []
  for f in os.listdir(path+'/' +folder):
    for line in open(path+'/'+folder+'/'+f):
      contador += 1
      print('%i/1041738' % contador,end='\r')
      files.append(ast.literal_eval(line.strip()))
      paths.append([path,folder,f])
      if int(files[-1]['likes']) not in total:
        total.append(int(files[-1]['likes']))
        c_value = int(files[-1]['likes'])
        
      # break
  total.sort()
  percentil = np.percentile(total,10)

  for d, p in zip(files, paths):
    texto = d['text'].strip().replace('\n','').strip()
    d_date = datetime.strptime(d['date'], "%d de %B de %Y")
    c_date = datetime.strptime(d['collect_date'], "%d de %B de %Y")

    if float(d['likes']) > percentil:
      try:
        # pos_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
        # f_out = "corpus/util/"+'/'.join(p[:-1])
        # if os.path.exists(f_out):
            # out = open(f_out+'/'+p[-1], 'w')
        if len(texto) > 0:
          class_file.write("%d, %d\n" % (d['id'], 1))
          cont_util+=1
        else:
          err_file.write(str(d['id'])+'\n')
      except:
        err_file.write(str(d['id'])+'\n')
    else:
        if (c_date - d_date).days <= 30:
          err_file.write(str(d['id'])+'\n')
          continue
        try:
          if len(texto) > 0:
            cont_nutil+=1
            class_file.write("%d, %d\n" % (d['id'], 0))
          else:
            err_file.write(str(d['id'])+'\n')
        except:
          err_file.write(str(d['id']))

print("Total Util: %d" % (cont_util))
print("Total Não Útil: %d" % (cont_nutil))
print("Total Geral: %d" % (cont_util+cont_nutil))

err_file.close()
class_file.close()

# exit()



# with open(sys.argv[1],'r') as input_file:
#       for line in input_file:
#         d = ast.literal_eval(line)
#         d['texto'] = d['texto'].strip().replace('\n','').strip()
#         d['id'] = str(d['id'])
#         d_date = d['data'].split(' ')
#         if int(d_date[0]) > 25 and d_date[-1] == '2019' and d_date[2] == 'março':
#             err_file.write(d['id']+'\n')
#             continue

#         if float(d['likes']) > 5:
#           try:
#             pos_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
#           except:
#             err_file.write(d['id'])
#         else:
#           try:
#             neg_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
#           except:
#             err_file.write(d['id'])
# neg_file.close()
# pos_file.close()
# err_file.close()
