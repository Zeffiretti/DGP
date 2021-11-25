###########################################################################
###                     Rat Interaction Data Loader                     ###
###                          ZEFFIRETTI, HIESH                          ###
###                   Beijing Institute of Technology                   ###
###                zeffiretti@bit.edu.cn, hiesh@mail.com                ###
###########################################################################


import pandas as pd


def format_header(file_path, output_file='') -> str:
    file_name = ''
    if file_path.endswith('.csv'):
        file_name = file_path[:-4]
    output_file_path = output_file if output_file else file_name+'-fo  rmat.csv'
    # print(output_file_path)

    header = pd.read_csv(file_path, header=None, index_col=False)
    print(type(header))
    print(header.loc[:2, 0:3])
    header.loc[2, :] = header.loc[0, :]+':'+header.loc[1, :]
    print(header.iloc[:2, 0:3])
    header.to_csv(output_file_path, index=False, header=False)

    return output_file_path


def load_int_data(file_path, mformat_header=True, header=2, index_col=0) -> pd.DataFrame:
    str = file_path if not mformat_header else format_header(file_path)
    return pd.read_csv(str, header=header, index_col=index_col)
    pass
