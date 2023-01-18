import pandas as pd

def post_dataframe_to_latex_table(df: pd.DataFrame,
                                  table_name: str,
                                  float_format: str = "%.3f",
                                  target_folder_path: str = '../../table/',
                                  kwargs: dict = {}
                                  ) -> None:
    """
    Output a dataframe to a LaTeX table.

    Parameters:
    -------
    df (pd.DataFrame): The dataframe to be output.

    table_name (str): The name of the LaTeX table.

    float_format (str, optional): The format of the floats. Defaults to "%.3f"

    target_folder_path (str, optional): The path to the folder where the
    table will be saved. Defaults to '../../table/'

    kwargs (dict, optional): Additional keyword arguments to pass to the latex command.
    
    Returns:
        None
    """
    df.to_latex(target_folder_path + table_name + '.tex', float_format = float_format, **kwargs)
    print('Save table to:{}'.format(target_folder_path))