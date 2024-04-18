import pandas as pd
import hgtk
import time
import datetime

def decompose(forms):
    jamo = []
    for form in forms:
        if hgtk.checker.is_hangul(form):
            word = ''

            for s in form:
                a, b, c = hgtk.letter.decompose(s)
                if not a:
                    a = '-'
                if not b:
                    b = '-'
                if not c:
                    c = '-'
                word = word + a + b + c
            jamo.append(word)
        else:
            jamo.append(form)
    return jamo

df_name = '../NIKL_MP_CSV/NXMP1902008040_{}.csv'
start = time.time()

with open('decomposed_sent.txt', 'a') as f:
    for i in range(5):
        df = pd.read_csv(df_name.format(i + 1))
        print(f'Now on : {df_name.format(i + 1)}')

        sentence_id_list = df['sentence_id'].unique()
        i = 0
        for sentence_id in sentence_id_list:
            i += 1
            forms = df[df['sentence_id']==sentence_id]['form'].values
            f.write(' '.join(decompose(forms)) + '\n')
            if i % 1000 == 0:
                print(f'{i} / {len(sentence_id_list)}')

end = time.time()
elapsed_time = datetime.timedelta(seconds=(end-start))

print(f'total elapsed time : {elapsed_time}') 
