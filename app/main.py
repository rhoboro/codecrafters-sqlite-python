import sys


from .sqlite import Database


database_file_path = sys.argv[1]
command = sys.argv[2]
database = Database(database_file_path)

if command == ".dbinfo":
    print(f"database page size: {database.header.page_size}")
    print(f"number of tables: {database.num_of_tables}")

elif command == ".tables":
    print(" ".join(f"{name}" for name, table in database.tables.items()))

elif command == ".print_schema":
    database.print_schema()

elif command == ".print_rows":
    database.print_rows()

else:
    print("\n".join(str(row) for row in database.query(command)))
