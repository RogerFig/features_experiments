import sys, ast, os
import numpy as np
#from scipy import stats
import json

class_file = open('filmow_utl.classes','w')
# pos_file = open('filmow_utl.pos','w')
err_file = open('filmow_utl.err','w')

path = sys.argv[1]
folders = os.listdir(path)

contador = 0
for folder in folders:
  total = []
  files = []
  paths = []
  for f in os.listdir(path+'/' +folder):
    for line in open(path+'/'+folder+'/'+f):
      contador += 1
      print('%i/1839851' % contador,end='\r')
      files.append(ast.literal_eval(line.strip()))
      paths.append([path,folder,f])
      if int(files[-1]['likes']) not in total:
        total.append(int(files[-1]['likes']))
        c_value = int(files[-1]['likes'])
        
      break
  if len(total) == 0: continue
  total.sort()
  try:
    percentil = np.percentile(total,10)
  except:
    print(folder)
    exit()
  for d, p in zip(files, paths):
    texto = d['text'].strip().replace('\n','').strip()
    
    if float(d['likes']) > percentil:
      try:
        if len(texto) > 0:
          class_file.write("%d, %d\n" % (d['id'], 1))
        else:
          err_file.write(str(d['id'])+'\n')
      except:
        err_file.write(str(d['id'])+'\n')
    else:
        
        if 'horas' in d['date'] or 'minutos' in d['date'] or '1 dia ' in d['date'] or ('dias' in d['date'] and int(d['date'].split(' ')[0]) < 5):
          err_file.write(str(d['id'])+'\n')
          continue
        try:
          # neg_file.write('0'*(18-len(d['id'])) + d['id'].strip() + ' ' + d['texto']+'\n')
          # f_out = "corpus/nao_util/"+'/'.join(p[:-1])
          # if os.path.exists(f_out):
            # out = open(f_out+'/'+p[-1], 'w')
          if len(texto) > 0:
            class_file.write("%d, %d\n" % (d['id'], 0))
          else:
            err_file.write(str(d['id'])+'\n')
          # out.write(json.dumps(d))
            # out.close()
          # else:
          #   os.mkdir(f_out)
          #   out = open(f_out+'/'+p[-1], 'w')
          #   out.write(json.dumps(d))
          #   out.close()
        except:
          err_file.write(str(d['id'])+'\n')

err_file.close()
class_file.close()