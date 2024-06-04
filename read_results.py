
from collections import defaultdict
import numpy as np
import scipy.stats as stats
def comparative_result_DID(f1=True,alpha=0.05, backbone='camel'):
    dataset=[]

    task_dic= {'Arabic Dialect Identification':[ 'Corpus-26', 'Corpus-6', 'Nadi'],'Sentiment Analysis': ['SemEval17','ASAD','AJGT','ASTD','LABR','ArSAS_Sentiment'], 'Hate Speech and Offensive Language Detection':['HateSpeech','Offensive', 'Adult']}
    # task_dic= {'Arabic Dialect Identification':['Corpus-2', 'Corpus-6', 'Corpus-9', 'Corpus-26', 'Nadi'],'Sentiment Analysis': ['SemEval17','ASAD','AJGT','ASTD','LABR','ArSAS_Sentiment'], 'Hate Speech and Offensive Language Detection':['HateSpeech','Offensive', 'Adult']}
    # for d in ['ASAD', 'SemEval17', 'AJGT', 'Corpus-2', 'Adult', 'Offensive', 'Corpus-26', 'ASTD', 'Corpus-6', 'HateSpeech', 'ArSAS_Sentiment', 'Corpus-9', 'LABR', 'Nadi']:
    results =open('results.txt').readlines()

    for i in range(len(results)):
        results[i]=results[i].replace('base-plm:/workspace/plm/', '').replace('task:', '').replace('plm:', '')

    # print(set([rs.split(',')[2] for rs in results]))
    # print(results[0].strip().split(','))
    # exit()
    baselines = ['mbert','arbert', 'mdbert', 'camel',  'mabert']


    for b in baselines:
        print(b.upper(),end=' ')
        for t in task_dic['Arabic Dialect Identification']:
            acc, f1 = [], []
            for line in results:
                line= line.replace(' ', '').strip().split(',')

                if t ==line[0] and backbone == line[1] and b == line[2]:
                    # acc.append(round(float(line[-1].split(':')[1])*100, 1))
                    acc.append(float(line[-1].split(':')[1]))
                    f1.append(float(line[-2].split(':')[1]))
            if len(acc) and len(f1):
                print('&{}$\pm${}&{}$\pm${}'.format(round(np.mean(acc)*100, 1), round(np.std(acc)*100, 2), round(np.mean(f1)*100, 1), round(np.std(f1)*100, 2)),  end='')
        print('\\\\')
    exit()
    for k, v in task_dic.items():
        if k not in ['Arabic Dialect Identification']: continue

        for d in v:

            # for b in baselines:

                # for res

            # if d !='Corpus-2':continue
            # if d in ['Corpus-2', 'Adult', 'Offensive', 'LABR']:continue
            # if d in ['Nadi']:continue
            # print(d)
            allres_=[]
            # for method in ['mbert','arbert','bashar','labse','camel','ours']:
            if k =='Arabic Dialect Identification':
                methds = ['mbert','labse','arabert','arbert','mdbert','camel','mabert','ours']
            else:
                methds = ['mbert','labse','arabert','arbert','mdbert','camel','ours','mabert']
            for method in methds:
            # for method in ['mbert','labse','arabert','arbert','camel','ours']:
            #for method in ['arbert','labse','camel','ours']:
                res=[]
                if method=='ours':
                    # pth='results/_results_ours_4_29.txt'.format(method)
                    pth='results/results_{}.txt'.format(method)
                else:
                    pth='results/results_{}.txt'.format(method)


                for line in open(pth).read().splitlines():
                    line = line.replace(',', '').split()
                    if line[1] ==d:
                        if f1:
                            res.append(float(line[8]))
                        else:
                            res.append(float(line[10]))
                allres_.append(res)
                # print(res)
            t_statistic, p_value = stats.f_oneway(allres_[0], allres_[1], allres_[2], allres_[3])
            # t_statistic, p_value = stats.f_oneway(allres_[0], allres_[1], allres_[2], allres_[3], allres_[4])
            #print(allres_)
            try:
                min_len = min([len(c) for c in allres_])
                allres_ = np.array([c[-min_len:] for c in allres_])
                max_ind = np.argmax(np.mean(allres_, axis=-1), axis=-1).item()
            except:
                print(allres_)
                print(d, method)
                continue
            #allres_ = np.array([c[-min_len:] for c in allres_])
            #max_ind = np.argmax(np.mean(allres_, axis=-1), axis=-1).item()
            if '_' in d:d=d.split('_')[0]
            rs=d.replace('Corpus', 'MADAR').upper()+''
            for i, c in enumerate(allres_):
                if i == max_ind:
                    rs += '& \\textbf{{{}}} $\pm$ \\textbf{{{}}}{}'.format(round(np.mean(c) * 100, 1),
                                                                           round(np.std(c) * 100, 1),
                                                                           '$^*$' if p_value < alpha else '')
                else:
                    rs += '& {} $\pm$ {}'.format(round(np.mean(c) * 100, 1), round(np.std(c) * 100, 1))
            # print(rs+'\\\\')
            print(rs + '\\\\')
        print()
    # print(set(dataset))
def comparative_result(f1=True,alpha=0.05):
    dataset=[]

    task_dic= {'Arabic Dialect Identification':['Corpus-2', 'Corpus-6', 'Corpus-9', 'Corpus-26', 'Nadi'],'Sentiment Analysis': ['SemEval17','ASAD','AJGT','ASTD','LABR','ArSAS_Sentiment'], 'Hate Speech and Offensive Language Detection':['HateSpeech','Offensive', 'Adult']}
    # for d in ['ASAD', 'SemEval17', 'AJGT', 'Corpus-2', 'Adult', 'Offensive', 'Corpus-26', 'ASTD', 'Corpus-6', 'HateSpeech', 'ArSAS_Sentiment', 'Corpus-9', 'LABR', 'Nadi']:
    for k, v in task_dic.items():
        print(f'\hline \hline  \multicolumn{{9}}{{c}}{{\\textbf{{{k}}}}}\\\\ \hline \hline ')

        for d in v:

            # if d !='Corpus-2':continue
            # if d in ['Corpus-2', 'Adult', 'Offensive', 'LABR']:continue
            # if d in ['Nadi']:continue
            # print(d)
            allres_=[]
            # for method in ['mbert','arbert','bashar','labse','camel','ours']:
            if k =='Arabic Dialect Identification':
                methds = ['mbert','labse','arabert','arbert','mdbert','camel','mabert','ours']
            else:
                methds = ['mbert','labse','arabert','arbert','mdbert','camel','ours','mabert']
            for method in methds:
            # for method in ['mbert','labse','arabert','arbert','camel','ours']:
            #for method in ['arbert','labse','camel','ours']:
                res=[]
                if method=='ours':
                    # pth='results/_results_ours_4_29.txt'.format(method)
                    pth='results/results_{}.txt'.format(method)
                else:
                    pth='results/results_{}.txt'.format(method)


                for line in open(pth).read().splitlines():
                    line = line.replace(',', '').split()
                    if line[1] ==d:
                        if f1:
                            res.append(float(line[8]))
                        else:
                            res.append(float(line[10]))
                allres_.append(res)
                # print(res)
            t_statistic, p_value = stats.f_oneway(allres_[0], allres_[1], allres_[2], allres_[3])
            # t_statistic, p_value = stats.f_oneway(allres_[0], allres_[1], allres_[2], allres_[3], allres_[4])
            #print(allres_)
            try:
                min_len = min([len(c) for c in allres_])
                allres_ = np.array([c[-min_len:] for c in allres_])
                max_ind = np.argmax(np.mean(allres_, axis=-1), axis=-1).item()
            except:
                print(allres_)
                print(d, method)
                continue
            #allres_ = np.array([c[-min_len:] for c in allres_])
            #max_ind = np.argmax(np.mean(allres_, axis=-1), axis=-1).item()
            if '_' in d:d=d.split('_')[0]
            rs=d.replace('Corpus', 'MADAR').upper()+''
            for i, c in enumerate(allres_):
                if i == max_ind:
                    rs += '& \\textbf{{{}}} $\pm$ \\textbf{{{}}}{}'.format(round(np.mean(c) * 100, 1),
                                                                           round(np.std(c) * 100, 1),
                                                                           '$^*$' if p_value < alpha else '')
                else:
                    rs += '& {} $\pm$ {}'.format(round(np.mean(c) * 100, 1), round(np.std(c) * 100, 1))
            # print(rs+'\\\\')
            print(rs + '\\\\')
        print()
    # print(set(dataset))


def read_res(method='ours'):
    res = defaultdict(list)
    for line in open('results_{}.txt'.format(method)).read().splitlines():
        line = line.replace(',', '').split()
        # if line[1]!='SemEval17':continue
        res[line[1]].append([float(line[8]),float(line[10])])
    for d, p in res.items():
        f1 = [f[0]  for f in p ]
        acc = [f[1]  for f in p ]
        # print('dataset {} f1 {}({}) acc {}({}) '.format(d, round(np.mean(f1), 4)*100,   round(np.std(f1),4)*100,  round(np.mean(acc),4)*100,  round(np.std(acc),4)*100))
        print('dataset {} f1 {} ({}) & {} ({}) '.format(d, round(np.mean(f1), 4)*100,   round(np.std(f1),4)*100,  round(np.mean(acc),4)*100,  round(np.std(acc),4)*100))
    # print(res)
if __name__ == '__main__':
    # read_res()
    # comparative_result()
    comparative_result_DID()
