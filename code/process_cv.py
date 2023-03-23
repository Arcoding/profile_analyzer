import os
from  preprocess_text import *
from unidecode import unidecode 
import re
import datetime
import shutil
sections_exps = {'experience':
                 [  
                     'PROFESSIONAL EXPERIENCE','WORK EXPERIENCE','EMPLOYMENT HISTORY','CAREER HISTORY',
                     'EXPERIENCIA PROFESIONAL',
                     'EXPERIENCE','EXPERIENCIA'
                 ],
                 'study':
                 [
                      'EDUCATION','EDUCACION','ESTUDIOS'
                 ],                 
                  'certificates':
                 [
                     'CERTIFICATES', 'CERTIFICATIONS', 'CERTIFICADOS','CERTIFICACIONES'
                 ],                 
                  'trainings':
                 [
                     'TRAININGS', 'COURSES', 'CAPACITACIONES'
                 ],                 
                  'skills':
                 [
                     'SKILLS','HABILIDADES'
                 ]
                 
                 
                }
def export_html(dict_sections, destination_path,template_path, css_path):
    output_file = os.path.join(destination_path,'output.html')
    shutil.copyfile(template_path, output_file) # Copy html template 
    shutil.copyfile(css_path, os.path.join(destination_path,'styles.css')) # Copy css 

    with open(output_file, mode ='r') as f:
        filedata = f.read()

    for k,v in dict_sections.items():
        filedata = filedata.replace(f'[{k}]', v)

    with open(output_file, mode ='w') as f:
        f.write(filedata)  
    print('Output file CREATED and saved as', output_file)      

def clean_skills(txt_content):
    txt_content = txt_content.lower().replace(' ','_').replace('-','_')
    return txt_content

def get_profile(df_concat):
    # Add Scores
    ## It contains the association between months of experience -> score
    df_concat['years_threshold'] = (df_concat['total_months']/12).round(1)
    df_concat['skill'] = df_concat['skill'].map(clean_skills)

    df_scores = pd.read_excel('../data/config.xlsx', sheet_name = 'scores')
    df_concat.sort_values(by = 'years_threshold', inplace=True)
    df_concat = pd.merge_asof(df_concat, df_scores, on = 'years_threshold' )
    del df_scores

    ## Add skill - Expertise
    df_sk_ex = pd.read_excel('../data/config.xlsx', sheet_name = 'expertise_skill')
    df_sk_ex['skill'] = df_sk_ex['skill'].map(clean_skills)
    df_concat = df_concat.merge(df_sk_ex, how = 'outer', on = 'skill' , validate = 'many_to_one')

    ## Add skill - role
    df_sk_role = pd.read_excel('../data/config.xlsx', sheet_name = 'skills_role')
    df_sk_role['skill'] = df_sk_role['skill'].map(clean_skills)
    df_concat = df_concat.merge(df_sk_role, how = 'left', on = 'skill' )

    ## Add skill - Expertise
    df_ex_role = pd.read_excel('../data/config.xlsx', sheet_name = 'expertise_role')
    df_concat = df_concat.merge(df_ex_role.rename(columns = {'weight':'weight_expertise'}), how = 'left', on = ['expertise_area','role' ])

    #Calculate final score
    df_concat['final_score'] = df_concat['Score'] * df_concat['weight'] *df_concat['weight_expertise'] 

    ## Summaryze
    output = df_concat.groupby(['role']).agg({'final_score':'sum'}).reset_index()
    print('Profile summarize CREATED')
    return output.sort_values(by='final_score', ascending=False)



## Cleaning text
## Removing special characterst and replacing date related values
def clean_text(txt_content):
    ## Replace -Present variations in text
    txt_content = re.sub(r'[ -]?\b(at)?present\b',' '+datetime.date.today().strftime(format = '%Y %m'), txt_content, flags=re.IGNORECASE)
    ## Replace -Present variations in text
    txt_content = re.sub(r'[ -][aA]ctual(idad)?',' '+datetime.date.today().strftime(format = '%Y %m'), txt_content)
    ## Replace , and -
    #txt_content = re.sub(r'[,-]'," ", txt_content)
    ## Remove duplicated spaces
    txt_content = txt_content.replace(' de ', ' ')
    
    ## This could be skipped when a new code for processing Spanish CV's is developed
    months_dict = {
        'enero':'January',
        'febrero':'February',
        'marzo':'March',
        'abril':'April',
        'mayo':'May',
        'junio':'June',
        'julio':'July',
        'agosto':'August',
        'septiembre':'September',
        'setiembre':'September',
        'octubre':'October',
        'noviembre':'November',
        'diciembre':'December',
        
    }
    for month,val in months_dict.items():
        txt_content = txt_content.lower().replace(month,val, )
        
    ## Remove accents
    txt_content =''.join(c for c in unicodedata.normalize('NFD', txt_content)   if unicodedata.category(c) != 'Mn')
    return txt_content

def process_one_cv(fname,cv_folder,results, nlp):
    output_folder = os.path.join('../data/output/', fname.replace('.pdf',''))
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)    
    print(f'Results will be stored at  {output_folder}')

    DIGITIZED_FILE = os.path.join(cv_folder, fname)
    ## Read PDF

    span_df = preprocess_text(DIGITIZED_FILE)

    ## Get useful information for identifying headers
    span_df = find_tags(span_df)

    ## Assign column class
    span_df = assign_columns(span_df)
    print('the file {} has {} columns in the document'.format(fname,span_df['column'].nunique() ))
    results['fname'].append(fname)
    results['n_columns'].append(span_df['column'].nunique())
    
    ## Sort values 
    span_df.sort_values(by=['column','page_num', 'ymin', 'xmin'], inplace=True)    

    ## Identify sections
    span_df = get_sections(span_df, sections_exps)

    if span_df['section'].nunique() >0:
        dict_sections = {}

        ## Start processing work history
        ## Keep only work experience section
        if 'experience' in span_df['section'].unique():
            
            df_exp = span_df[span_df['section']=='experience'].sort_values(by=['page_num','ymin'])
            df_exp['text'] = df_exp['text'].map(clean_text)

            ## Define patterns 
            exps = [
                r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)\D?(\d{1,2}(st|nd|rd|th)?)?(([,.-/])\D?)?,?(\s)?((19[7-9]\d|20\d{2})|\d{2})',
                r'\b(?:Ene(?:ro)?|Feb(?:rero)?|Mar(?:zo)?|Abr(?:il)?|May(?:o)?|Jun(?:io)?|Jul(?:io)?|Ago(?:sto)?|Sep(?:tiembre)?|Set(?:iembre)?|Oct(?:ubre)?|(Nov|Dic)(?:iembre)?)\D?(\d{1,2}(st|nd|rd|th)?)?(([,.-/])\D?)?,?(\s)?((19[7-9]\d|20\d{2})|\d{2})\b',
                
                r'\b(\d{2} )?(\d{2} )(20\d{2})\b(?!\d)',
                r'\b(\d{2}\/)?(\d{2}\/)(20\d{2})\b(?!\d)',
                r'\b(\d{2}\-)?(\d{2}\-)(20\d{2})\b(?!\d)',
                r'\b(\d{2}\.)?(\d{2}\.)(20\d{2})\b(?!\d)',
                r'\b(\d{2})?(\d{2})(20\d{2})\b(?!\d)',
                
                r'\b(20\d{2} )(\d{2})( \d{2})?\b',
                r'\b(20\d{2}\/)(\d{2})(\/\d{2})?\b',
                r'\b(20\d{2}\-)(\d{2})(\-\d{2})?\b',
                r'\b(20\d{2}\.)(\d{2})(\.\d{2})\b'
            ]
            dates_df = find_dates(df_exp, exps)
            dates_df = match_pair_dates(dates_df)
            ## Return dates to their original lines
            df_exp = df_exp.merge(dates_df, how = 'left', validate='many_to_one', left_index = True, right_index = True )   

            # Get entities using NLP
            df_exp['entities'] = df_exp['text'].apply(get_ents, nlp = nlp) 


            ## Get useful columns from entities
            for label in nlp.get_pipe("entity_ruler").labels:
                df_exp[label] = df_exp['entities'].map(lambda x: x[label])
                df_exp[label + '_FOUND'] = df_exp['entities'].map(lambda x: len(x[label])>0)

            ## Process job title
            df_exp = assign_job_title(df_exp)

            # Get skills
            df_skills = get_skills(df_exp)
            df_skills = df_skills[(df_skills['start_date'].notnull()) & (df_skills['end_date'].notnull())]
            # Clear skills
            if df_skills.shape[0]>0:

                df_skills = clear_skills(df_skills)


                # Save df_skills
                df_skills.to_csv(os.path.join(output_folder, 'df_skills.csv'), index=False)
                dict_sections['SKILLS'] = (
                    df_skills
                    .to_html(index=False, classes='mystyle'))

                # Calculate profile
                df_profile = get_profile(df_skills)
                dict_sections['SUMMARY'] = (
                    df_profile
                    .to_html(index=False, classes='mystyle'))                

                print('Skills SAVED...')
            else:
                print('SKILLS NOT FOUND!')

            # Get Roles
            df_roles = df_exp[(df_exp['start_date'].notnull()) & (df_exp['end_date'].notnull()) ][['start_date','end_date', 'JOB_TITLE']].drop_duplicates()
            if df_roles.shape[0]>0:
                df_roles = get_work_hist(df_roles)


                #Save df_skills
                df_roles.to_csv(os.path.join(output_folder, 'df_roles.csv'), index=False)  
                dict_sections['EXPERIENCE'] = (
                    df_roles
                    .rename(columns = {'duration':'months'})
                    .drop(columns = ['count_axis'])
                    .to_html(index=False, classes='mystyle'))




                print('Roles SAVED...')
            else:
                print('JOB TITLES AND DATES NOT FOUND')
        else:
            print('WORK HISTORY SECTION NOT FOUND!')

 
        export_html(
                dict_sections, 
                destination_path = output_folder,
                template_path = 'test.html',
                css_path = 'styles.css'
                 )
    else:
        print('SECTIONS NOT FOUND!')
    print('*'*50)
    print('\n')
