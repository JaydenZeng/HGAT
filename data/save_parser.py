import numpy as np
def save_parser_data(readpath, savepath):
   with open(readpath, 'r+') as fr:
      lines = fr.readlines()
      print (len(lines))
      res = []
      cur = []
      for line in lines:
         if len(line.strip('\n')) != 0:
            cur.append(line.strip('\n'))
         else:
            res.append(cur)
            cur = []
   np.save(savepath, res)
#save_parser_data('./beauty/beauty-que-parse.output', './beauty/beauty-que-parse.npy')
