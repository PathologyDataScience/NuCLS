import sqlite3


def list_to_str(input_list, delim=','):
    """
    Convert list to delimited string
    """
    output_string = input_list[0]
    for i, j in enumerate(input_list):
        if i == 0:
            continue
        output_string = output_string + delim + j

    return output_string


class SQLite_Methods(object):
    """
    This class handles interactions with the histomics TK girder API
    """
    
    def __init__(self, db_path=None):    
        
        # Initialize connection
        # =====================================================================
        
        if db_path is None:
            input("No database connected, press ENTER to connect to RAM")
            db_path = ":memory"
        
        self.db_path = db_path        
        self.conn = sqlite3.connect(self.db_path)
        self.c = self.conn.cursor()
        
    #%%============================================================================
    # General SQL connection methods
    #==============================================================================
    
    def get_existing_tables(self):
        """
        get list of existing tables in database
        """
        self.c.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        table_list = self.c.fetchall()
        return [j[0] for j in table_list]
            
    #%%============================================================================
    
    def commit_changes(self):
        """
        commit changes - can ignore this if using "with self.conn"
        """
        self.conn.commit()
       
    #%%============================================================================
    
    def close_connection(self):
        """
        commit changes - can ignore this if using "with self.conn"
        """
        self.c.close()
    
    #%%============================================================================
    
    def get_table_headers(self, table_name):
        """
        Get table headers (column names)
        """
        self.c.execute("PRAGMA TABLE_INFO('%s')" % (table_name))
        return [j[1] for j in self.c.fetchall()]
    

    #%%============================================================================
    # Methods to clean up and parse DF's for SQL compatibility
    #==============================================================================
    
    def parse_colnames_for_sql(self, colnames, bad_patterns=None, bad_col_names=None):
        """
        Parses column names to be SQL-compatible
        """
        
        if bad_patterns is None:
            bad_patterns = (' ', '(', ')')
        if bad_col_names is None:    
            bad_col_names = ("group", "Group", "id")
        
        for i,j in enumerate(colnames):
            for patt in bad_patterns:
                if patt in j:
                    colnames[i] = colnames[i].replace(patt, '_')
            if j in bad_col_names:
                colnames[i] += '_'
        return colnames
    
    #%%============================================================================
        
    def remove_repeated_rows_by_column(self, data_frame, colname):
            """
            Given a pandas dataframe, remove repeated rows at a specific column
            """
            existing_ids = []    
            for annid, ann in data_frame.iterrows():
                if ann[colname] in existing_ids:
                    data_frame.drop(annid, axis=0, inplace=True)
                existing_ids.append(ann[colname])
            data_frame.reset_index(inplace=True)
            data_frame.drop("index", axis=1, inplace=True)
            
            return data_frame
            
    #%%============================================================================
    
    def parse_dframe_to_sql(self, dframe, primary_key):
        """
        Parse pandas dataframe to generate SQL-friendly string commands
        """
        varnames = list(dframe.columns)    
        
        # Getting stypes - handling NAN values (which show as "float")
        #
        # Get unique type per column
        dtypeCount =[dframe.iloc[:,i].apply(type).value_counts() for i in range(dframe.shape[1])]
        dtypes = [[k.__name__ for k in list(j.index)] for j in dtypeCount]
        
        # if NAN is there (neside some other type), remove it
        for i,j in enumerate(dtypes):
            if len(j) > 1 and 'float' in j:
                dtypes[i].remove('float')
        dtypes = [j[0] for j in dtypes]
        
        # replace by equivalent SQlite data types
        for i,j in enumerate(dtypes):
            if 'str' in j:
                dtypes[i] = 'text'
            elif 'float' in j:
                dtypes[i] = 'real'
            elif 'int' in j:
                dtypes[i] = 'integer'
            elif 'bool' in j:
                dtypes[i] = 'integer'
            else:
                dtypes[i] = 'text'
            
            # add primary key    
            if primary_key == varnames[i]:
                dtypes[i] += ' NOT NULL PRIMARY KEY'
        
        # get string for "create table" SQLite command
        create_str = [j + ' ' + dtypes[i] for i,j in enumerate(varnames)]
        create_str = list_to_str(create_str, delim=', ')
        
        # get string to update a column from the table
        update_str = [':' + j for j in varnames]
        update_str = list_to_str(update_str, delim=', ')
        
        sql_strings = {'varnames': varnames,
                       'dtypes': dtypes,
                       'create_str': create_str,
                       'update_str': update_str,}
        
        return sql_strings
            
    #%%============================================================================
    #  Methods to add tables/data to SQL tables   
    #==============================================================================
    
    def create_sql_table(self, tablename, create_str):
        """
        create SQLite table using create string
        """
        self.c.execute("CREATE TABLE IF NOT EXISTS {} ({})".format(tablename, create_str))
    
    #%%============================================================================
    
    def update_sql_table(self, tablename, entry, update_str):
        """
        update SQLite table using update string and pd series entry, indexed by varname
        """
        
        # replace NA with 0
        entry.fillna(value=0, inplace=True)
    
        insert_str = "INSERT INTO {} VALUES ({})".format(tablename, update_str)
        self.c.execute(insert_str, dict(entry))
    
    #%%============================================================================
    # Methods to fetch data from SQL tables
    #==============================================================================
    
    def fetch_identifiers(self, tablename, varname):
        """
        fetch unique identifiers from table
        """
        self.c.execute("SELECT {} FROM {}".format(varname, tablename))
        identifiers = self.c.fetchall()
        return [j[0] for j in identifiers]




#%% ###########################################################################
#%% ###########################################################################
#%% ###########################################################################
