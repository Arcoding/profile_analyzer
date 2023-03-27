from unidecode import unidecode 
import fitz
import re
import numpy as np
import pandas as pd
from spacy import displacy
import spacy
import unicodedata

def get_general_info(span_df):
    contact_info = {}

    ## First find name
    name = span_df[
        (span_df['font_size'] ==span_df['font_size'].max()) 
        & (span_df['page_num']==1)
        & (span_df['ymax']<=200) ]['text']
    name = ''.join(name)    
    if len(name)>0:
        contact_info['Name'] = name

    ## Find other information
    text = ' '.join(span_df[span_df['page_num']==1]['text'].values).lower() ## this should be in first page
    patterns = {
    "email":[r'[a-z0-9\.\-+_]+@[a-z0-9\.\-+_]+\.[a-z]+'],
    "phone" : [r'(?:\d{8}(?:\d{2}(?:\d{2})?)?|\(\+?\d{2,3}\)\s?(?:\d{4}[\s*.-]?\d{4}|\d{3}[\s*.-]?\d{3}|\d{2}([\s*.-]?)\d{2}\1\d{2}(?:\1\d{2})?))'] ,
    'linkedIn': [r'((?:https?:)?\/\/(?:[\w]+\.)?linkedin\.com\/in\/(?P<permalink>[\w\-\_À-ÿ%]+)\/?)']
    }
    for k,v in patterns.items():
        for patt in v:
            for match in re.finditer(patt, text):
                start, end = match.span()
                span = text[start: end]
                if len(span)>0:
                    contact_info[k] = span
                    break # just save first coincidence
            if len(span)>0:
                break

    return contact_info

def read_pdf_as_dict(DIGITIZED_FILE):

    ## Read pdf file with fitz
    print('Reading {}...'.format(DIGITIZED_FILE))
    with fitz.open(DIGITIZED_FILE) as doc:
        block_dict = {}
        page_num = 1
        line_num_test = 1

        for page in doc: # Iterate all pages in the document
            file_dict = page.get_text('dict') # Get the page dictionary 
            block = file_dict['blocks'] # Get the block information

            for a in block:   
                if a["type"] == 0:
                    for line in a['lines']:
                        for span in line['spans']:
                            span["page_num"] = page_num
                            span["line_num"] =  line_num_test
                            #print(span['text'])
                        line_num_test += 1

                else :
                    a["page_num"] = page_num

            block_dict[page_num] = block # Store in block dictionary

            page_num += 1 # Increase the page value by 1
            #print(f'processed page {page_num}...')
    print('reading PDF FINISHED')
    return block_dict


def extract_text_features(block_dict):
    rows = []
    ## Extract useful information for further processing steps
    #print('Converting Text to dictionary with useful features...')
    for page_num, blocks in block_dict.items():
        for block in blocks:
            if block['type'] == 0:
                for line in block['lines']:
                    for span in line['spans']:
                        xmin, ymin, xmax, ymax = list(span['bbox'])
                        font_size = span['size']
                        color = span['color']
                        text = ''.join(c for c in unicodedata.normalize('NFD', span['text'])   if unicodedata.category(c) != 'Mn')#unidecode(span['text'])
                        span_font = span['font']
                        num_page = span["page_num"]
                        line_num = span['line_num']
                        block_num = block['number']
                        is_upper = False
                        is_bold = False 
                        if "bold" in span_font.lower():
                            is_bold = True 
                        if re.sub("[\(\[].*?[\)\]]", "", text).isupper():
                            is_upper = True
                        if text.replace(" ","") !=  "":
                            rows.append((xmin, ymin, xmax, ymax, line_num, block_num, num_page, text, is_upper, is_bold, span_font, font_size, color))    
    print('Dictionary CREATED...')
    return rows

def get_df_from_text(rows):
    span_df = pd.DataFrame(rows, columns=['xmin','ymin','xmax','ymax', 'line_num', 'block_num', "page_num", 'text', 'is_upper','is_bold','span_font', 'font_size', 'color'])
    print('Dataframe CREATED...')
    return span_df

def preprocess_text(DIGITIZED_FILE):
    dict_content = read_pdf_as_dict(DIGITIZED_FILE)

    list_rows = extract_text_features(dict_content)
    return get_df_from_text(list_rows) 


def find_tags(span_df):    
    
    span_scores = []
    span_num_occur = {}
    special = '[(_:/,#%\=@)]'

    ## Get type of text
    ### In this section we will assign a class to each text
    ### 'p' will be the lowest type, commonly used for paragraph text
    ### Above 'p' will be 'h1', 'h2', etc.. usually headers of sections
    for index, span_row in span_df.iterrows():
        score = round(span_row.font_size)
        text = span_row.text

        if not re.search(special, text):
            if span_row.is_bold:
                score +=1 

            if span_row.is_upper:
                score +=1
        span_scores.append(score)
    values, counts = np.unique(span_scores, return_counts=True)

    values, counts = np.unique(span_scores, return_counts=True)
    style_dict = {}
    for value, count in zip(values, counts):
        style_dict[value] = count
    sorted(style_dict.items(), key=lambda x: x[1])

    p_size = max(style_dict, key=style_dict.get)
    idx = 0
    tag = {}

    for size in sorted(values, reverse = True):
        idx += 1
        if size == p_size:
            idx = 0
            tag[size] = 'p'
        if size > p_size:
            tag[size] = 'h{0}'.format(idx)
        if size < p_size:
            tag[size] = 's{0}'.format(idx)

    span_tags = [tag[score] for score in span_scores]

    span_df['tag'] = span_tags

    return span_df

def combine_lines(span_df, col_name ='line_num' ):
    #As can be seen in the dataframe, there is text from the same line separated by rows. This for/while loop joins them together with its correct bbox values per variable.
    ## Merge text from same line
    deletion = []

    span_df = span_df.groupby(['page_num','block_num','line_num']).agg({
        'xmin':'min',
        'ymin':'min',
        'xmax':'max',
        'ymax':'max',
        'text': lambda x: ' '.join(x.str.strip()),
        'is_bold':'max',
        'font_size':'max',
        'tag':'min'

    })

    span_df.reset_index(inplace = True)
    return span_df

def clear_duplicated_spaces(span_df):
    span_df['collapse'] = span_df['text'].str.replace(r' +',' ', regex=True).str.contains('[A-Z0-9]{1,2} [A-Z0-9]{1,2} [A-Z0-9]{1,2} [A-Z0-9]{1,2}', regex=True)
    span_df.loc[span_df['collapse'],'text']=span_df[span_df['collapse']]['text'].str.replace('  ','__').str.replace(' ','').str.replace('__',' ') 
    return span_df

def get_main_col_coordinates(page_one):
    page_one_width = page_one['xmax'].max() -  page_one['xmin'].min() # Get maximum range 
    page_x_min = page_one['xmin'].min()
    page_x_max = page_one['xmax'].max()
    page_x_mid = (page_x_min +  page_x_max)/2   

    ## Get potential rectangle occupied by main column
    main_col_candidates = page_one.groupby(['xmin']).agg({'line_num':'count' , 'xmax':'max','ymin':'min' }).reset_index()
    main_col_candidates.loc[main_col_candidates['xmin']<page_x_mid, 'xmin'] = page_x_min
    main_col_candidates.loc[main_col_candidates['xmin']>=page_x_mid, 'xmax'] = page_x_max
    main_col_candidates['width'] = main_col_candidates['xmax'] - main_col_candidates['xmin']
    tolerance = 10
    main_col = (
        main_col_candidates[main_col_candidates['width']+tolerance>page_one_width/2]
        .sort_values(by = 'line_num', ascending = False)
        .head(1)
        )
    
    ## Find coordinates occupied
    main_col_x_min = main_col['xmin'].values[0]
    main_col_x_max = main_col['xmax'].values[0]
    main_col_y_min = main_col['ymin'].values[0]

    return main_col_x_min, main_col_x_max, main_col_y_min

def get_sec_col_coordinates(page_one, main_col_x_min, main_col_x_max,main_col_y_min, sep_thresh=7):
    ## Find lines that do not overlap main column with threshold
    page_one['flg_other_column']=((page_one['xmax']+sep_thresh<main_col_x_min )| (page_one['xmin']>main_col_x_max+sep_thresh ))*1
    page_one['width'] = page_one['xmax']  - page_one['xmin']
    sec_col = page_one[(page_one['flg_other_column']==1 ) & (page_one['ymax']>=main_col_y_min)]
    
    ## Get the maximum xaxis coordinate of secondary column
    sec_col = sec_col[sec_col['width']==sec_col['width'].max()] 
    sec_col_x_min = sec_col['xmin'].min()
    sec_col_x_max = sec_col['xmax'].max()
    
    ## Recalculate coordinates of secondary column
    if sec_col_x_min<(page_one['xmax'].max() +  page_one['xmin'].min())/2:
        sec_col_x_min=0
    else:
        sec_col_x_max = page_one['xmax'].max()   

    return sec_col_x_min, sec_col_x_max


def assign_columns(span_df):
    span_df = combine_lines(span_df)
    span_df = clear_duplicated_spaces(span_df) ## better to read entire line to detect duplicated spaces
    page_one = span_df[span_df['page_num']==1]
    main_col_x_min, main_col_x_max, main_col_y_min = get_main_col_coordinates(page_one)
    sep_thresh = 7
    sec_col_x_min, sec_col_x_max = get_sec_col_coordinates(page_one, main_col_x_min, main_col_x_max,main_col_y_min, sep_thresh)

    span_df['column']='main'
    
    #Assign 'secondary'column tag if found
    lines_other_column = page_one['flg_other_column'].sum() ## count total 'secondary' lines
    if lines_other_column>=5: # Enough to conclude there are 2 columns
        span_df.loc[((span_df['xmin']>=sec_col_x_min - sep_thresh) & (span_df['xmax']<=sec_col_x_max+sep_thresh) ),'column']='secondary'
        
    return span_df


## Functions for finding Sections headers
def sections_finder(_df0, sections_dict, sufix ='section'):
    results = pd.DataFrame()
    
    _df = _df0[(_df0['tag']!='p') | (_df0['is_bold'])][['text','font_size','tag','line_num','page_num','column']]
    for section,patts in sections_dict.items():
        for patt in patts:
            sections_found = _df[_df['text'].str.lower().str.contains(patt.lower())]
            if sections_found.shape[0]>0:
                sections_found['section']= section
                sections_found['patt'] = patt
                sections_found['patt_len'] = len(patt.split(' '))
                results = pd.concat([results,sections_found], axis=0, ignore_index=True)
    
    if results.shape[0]>0:
        results.sort_values(by=['section', 'font_size','tag','patt_len', 'page_num','line_num'], ascending=[False, False,True,False,True,True], inplace=True)
        results.drop_duplicates(['section'],inplace=True, keep='first')
        results.sort_values(by = 'line_num', inplace=True)
    return results


def get_sections(span_df, sections_exps):
    df_sections = sections_finder(span_df, sections_exps)
    if df_sections.shape[0]>0:
        span_df  = (
            span_df
            .merge(
                df_sections[['line_num','page_num','section']],
                how = 'left',
                on = ['line_num','page_num'], 
                validate = 'many_to_one')
        )
        ## Fill sections column
        span_df['section']=span_df.groupby(['column'])['section'].ffill()
        print('Identified {} sections: {}'.format(df_sections.shape[0],df_sections['section'].unique() ))

    return span_df

def find_dates(df_exp, exps):
    ## Collect useful data from date text
    dates_dict = {
        'span_start':[],
        'span_end': [], 
        'span_text':[], 
        'date_formated':[],
        'idx':[]
    }

    ## process each pattern
    for idx,row in df_exp.iterrows():
        for reg in exps:
            for match in re.finditer(reg, row['text'], flags=re.IGNORECASE):
                start, end = match.span()
                span = row['text'][start: end]
                # This is a Span object or None if match doesn't map to valid token sequence
                if span is not None:
                    ## Save data from date matched
                    dates_dict['span_start'].append(start)
                    dates_dict['span_end'].append(end)
                    dates_dict['span_text'].append(span)
                    dates_dict['date_formated'].append(pd.to_datetime(span))
                    dates_dict['idx'].append(idx)   

    ## Create a dataframe from dates found
    dates_df = (
        pd.DataFrame(dates_dict)
        .sort_values(by ='span_start', ascending=True)
        .drop_duplicates(['span_start','span_end','idx']))
    return dates_df

def match_pair_dates(dates_df):
    dates_df = (
        dates_df
        .groupby('idx')
        .agg({'span_text':'count', 'date_formated':['min','max']})
        )

    dates_df.columns = ['n_dates', 'start_date', 'end_date']
    dates_df = dates_df.reset_index()
    #print(dates_df)
    dates_df['flg_pair_dates'] = False

    ## 2 dates in same line
    dates_df = dates_df[dates_df['n_dates']<=2]

    if dates_df.shape[0]>0:

        dates_df.loc[dates_df['n_dates']==2, 'flg_pair_dates'] = True 

        dates_df['flg_first' ] = False
        dates_df.loc[dates_df['n_dates']==2, 'flg_first'] = True 


        ## start_date found line before
        dates_df['start_date_lb'] = dates_df['start_date'].shift(1)
        dates_df['idx_lb'] = dates_df['idx'].shift(1)
        dates_df['flg_date_found_lb'] = (~dates_df['flg_pair_dates']) &  (dates_df['end_date']>dates_df['start_date_lb']) & (dates_df['idx'] - dates_df['idx_lb'] <=3 )## 3 is the lines threshold
        dates_df.loc[dates_df['flg_date_found_lb'],'start_date' ] = dates_df[dates_df['flg_date_found_lb']]['start_date_lb']
        dates_df.loc[(dates_df['start_date'].notnull()) &(dates_df['end_date'].notnull()) & (dates_df['end_date']!= dates_df['start_date']) ,'flg_pair_dates' ] = True

        dates_df['end_date_la'] = dates_df['end_date'].shift(-1)
        dates_df.loc[(dates_df['flg_date_found_lb'].shift(-1)).fillna(False),'end_date' ] = dates_df[(dates_df['flg_date_found_lb'].shift(-1)).fillna(False)]['end_date_la']
        dates_df.loc[(dates_df['flg_date_found_lb'].shift(-1)).fillna(False),'flg_first' ] = True

        ## start_date found line after
        dates_df['start_date_la'] = dates_df['start_date'].shift(-1)
        dates_df['idx_la'] = dates_df['idx'].shift(-1)
        dates_df['flg_date_found_la'] = (~dates_df['flg_pair_dates']) &  (dates_df['end_date']>dates_df['start_date_la']) & (dates_df['idx_la'] - dates_df['idx'] <=3 )## 3 is the lines threshold
        dates_df.loc[dates_df['flg_date_found_la'],'start_date' ] = dates_df[dates_df['flg_date_found_la']]['start_date_la']
        dates_df.loc[(dates_df['start_date'].notnull()) &(dates_df['end_date'].notnull()) & (dates_df['end_date']!= dates_df['start_date']) ,'flg_pair_dates' ] = True
        dates_df.loc[dates_df['flg_date_found_la'],'flg_first' ] = True

        dates_df = dates_df[dates_df['flg_pair_dates']] ## Keep only when pair found
        dates_df.set_index('idx', inplace=True)
        dates_df.index.name = None ## Remove index name
    
    return dates_df

## Find entities in text
def get_ents(text, nlp):


    nlp_text = nlp(text)
    entities = nlp_text.ents
    output = {}
    for label in nlp.get_pipe("entity_ruler").labels:
        output[label]= []
    for ent in entities:
            output[ent.label_].append(str(ent))
    return output

def assign_job_title(df_exp, col_name = 'JOB_TITLE'):
    # Assign Job Title if found

    ## Get first job title found
    df_exp.loc[df_exp['JOB_TITLE_FOUND'], col_name] = df_exp[df_exp['JOB_TITLE_FOUND']][col_name].map(lambda x: x[0])
    df_exp.loc[~df_exp['JOB_TITLE_FOUND'], col_name] = None

    ## Assign start and end date when job title found
    ### Look one line before
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['n_dates'].shift(1).notnull()) & (df_exp[col_name].shift(1).isnull()) , 'start_date']= df_exp['start_date'].shift(1)
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['n_dates'].shift(1).notnull()) & (df_exp[col_name].shift(1).isnull()), 'end_date']= df_exp['end_date'].shift(1)

    ## Look one line after
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['start_date'].isnull()) & (df_exp['n_dates'].shift(-1).notnull()) & (df_exp[col_name].shift(-1).isnull()) , 'start_date']= df_exp['start_date'].shift(-1)
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['end_date'].isnull()) & (df_exp['n_dates'].shift(-1).notnull()) & (df_exp[col_name].shift(-1).isnull()), 'end_date']= df_exp['end_date'].shift(-1)
    
    ## Look two line after
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['start_date'].isnull()) & (df_exp['n_dates'].shift(-2).notnull()) & (df_exp[col_name].shift(-2).isnull()) , 'start_date']= df_exp['start_date'].shift(-2)
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['end_date'].isnull()) & (df_exp['n_dates'].shift(-2).notnull()) & (df_exp[col_name].shift(-2).isnull()), 'end_date']= df_exp['end_date'].shift(-2)

    ## Remove job titles not close to dates
    df_exp.loc[(df_exp[col_name].notnull()) & (df_exp['start_date'].isnull()), col_name] = np.NaN

    ## Return entire text when role not found
    df_exp_tmp = df_exp[(df_exp['start_date'].notnull()) & (df_exp['end_date'].notnull())][['start_date','end_date',col_name]].drop_duplicates()
    df_exp_tmp['counter'] = df_exp_tmp.count(axis=1)
    df_exp_tmp = df_exp_tmp.sort_values(by = 'counter', ascending = False).drop_duplicates(['start_date','end_date'], keep='first')
    df_exp_tmp = df_exp_tmp[df_exp_tmp[col_name].isnull()]
    df_exp['has_dates_but_role'] = False
    df_exp.loc[df_exp_tmp.index, 'has_dates_but_role'] = True

    #df_exp['has_dates_but_role'] = (df_exp[col_name].isnull()) & (df_exp['start_date'].notnull()) & (df_exp['end_date'].notnull()) 
    df_exp.loc[df_exp['has_dates_but_role'] , col_name] =  ''
    font_size_mode = df_exp['font_size'].value_counts().index[0] ## Most frequent font_size
    for i in [2,1,0,-1]:
        df_exp['add_text_to_role'] = (
            (df_exp['has_dates_but_role']) ## has dates but role not found
            & ((df_exp['start_date'].shift(i).isnull())| (df_exp['start_date']==df_exp['start_date'].shift(i))) ## row doesn't have dates or belong to same work experience, not to other
            &  (df_exp['text'].shift(i).notnull()) ## there is a text
            &  (df_exp['flg_first']) ## there is a text
            &  ((df_exp['is_bold'].shift(i) )| (df_exp['font_size'].shift(i)>font_size_mode) | (i==0)) ## text is bold or font size is greater than the frequent value
            )
        df_exp['text_to_add'] = df_exp['text'].shift(i)
        df_exp.loc[df_exp['add_text_to_role'], col_name] =    df_exp[df_exp['add_text_to_role']][col_name]  + ' '+   df_exp[df_exp['add_text_to_role']]['text_to_add'] 

    ## Fill start and end dates, asumming date always are before the description of the experience 
    df_exp['start_date'] = df_exp['start_date'].ffill()
    df_exp['end_date'] = df_exp['end_date'].ffill()  
    df_exp.loc[df_exp[col_name].notnull(), col_name] = df_exp[df_exp[col_name].notnull()][col_name].map( lambda x: unidecode(x))         
    return df_exp

def get_skills(df_exp):
    ## Summary of Skills:
    dict_skills = {
        'skill':[],
        'start_date': [],
        'end_date':[]
    }
    df_skills = df_exp[df_exp['SKILL_FOUND']][['SKILL','start_date', 'end_date']]
    for _,row in df_skills.iterrows():
        for skill in row['SKILL']:
            dict_skills['skill'].append(skill)
            dict_skills['start_date'].append(row['start_date'])
            dict_skills['end_date'].append(row['end_date'])
    df_skills = pd.DataFrame(dict_skills).drop_duplicates()
    return df_skills    

def clear_skills(df_skills):
    # Remove traslaping dates
    df_skills_clean = pd.DataFrame()
    for _,i in df_skills.iterrows():
        df_to_add = pd.DataFrame(pd.date_range(i['start_date'],i['end_date'] ,freq="MS" ,inclusive='both' ))
        df_to_add['skill'] = i['skill']
        df_skills_clean = pd.concat([df_skills_clean,df_to_add], axis = 0, ignore_index=True)
        
    df_skills_clean.drop_duplicates(inplace=True)
    df_skills_clean = df_skills_clean.groupby(['skill']).agg({0:['count','min', 'max']}).reset_index()
    df_skills_clean.columns = ['skill','total_months','min_date', 'max_date']
    return df_skills_clean

def get_work_hist(df_roles):
    df_roles['count_axis'] = df_roles.count(axis=1)
    df_roles.sort_values(by = ['start_date','end_date','count_axis' ], ascending = False)
    df_roles.drop_duplicates(['start_date','end_date'], keep = 'first', inplace= True)
    df_roles['JOB_TITLE'] =df_roles['JOB_TITLE'].fillna('Unknown')
    df_roles['duration'] = (df_roles['end_date'] - df_roles['start_date']).dt.days // 30

    # Calculate total experience
    total_months = pd.DataFrame()
    for _,i in df_roles.iterrows():
        total_months = pd.concat([
            total_months,
            pd.DataFrame(pd.date_range(i['start_date'] - pd.Timedelta(days = i['start_date'].day-1),i['end_date'] ,freq="MS" ,inclusive='both' ))
        ], axis = 0, ignore_index=True)
    total_months.drop_duplicates(inplace=True)
    df_roles['start_date'] = df_roles['start_date'].dt.strftime('%m/%Y')
    df_roles['end_date'] = df_roles['end_date'].dt.strftime('%m/%Y')

    #print('First and Last months of experience:', total_months[0].min(), ' - ', total_months[0].max())
    #print('Total months of experience: {}'.format(total_months.shape[0]))    
    return df_roles

