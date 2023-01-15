import sys

from .sqlite import Database

# import sqlparse - available if you need it!


database_file_path = sys.argv[1]
command = sys.argv[2]
database = Database(database_file_path)

if command == ".dbinfo":
    print(f"database page size: {database.header.page_size}")
    print(f"number of tables: {database.num_of_tables}")

elif command == ".tables":
    print(" ".join(database.table_names))

elif command == ".print_rows":
    database.print_rows()

else:
    print(database.execute(command))
