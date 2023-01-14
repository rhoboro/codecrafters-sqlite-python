import sys

# import sqlparse - available if you need it!

from sqlite import Database

database_file_path = sys.argv[1]
command = sys.argv[2]

if command == ".dbinfo":
    database = Database(database_file_path)
    print(f"database page size: {database.header.page_size}")
    print(f"number of tables: {database.num_of_tables}")

elif command == ".tables":
    database = Database(database_file_path)
    print(" ".join(database.table_names))

else:
    print(f"Invalid command: {command}")
