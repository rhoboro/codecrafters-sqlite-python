import sys
from typing import Optional

from dataclasses import dataclass
from struct import unpack

HEADER_SIZE = 100


# import sqlparse - available if you need it!
@dataclass(frozen=True)
class FileHeader:
    magic_number: str
    page_size: int
    write_format: int
    read_format: int
    unused_reserved_space: int
    maximum_embedded_payload_fraction: int
    minimum_embedded_payload_fraction: int
    leaf_payload_fraction: int
    file_change_counter: int
    in_header_database_size: int
    first_freelist_trunk_page: int
    freelist_pages: int
    schema_cookie: int
    schema_format_number: int
    default_page_cache_size: int
    largest_root_btree_page: int
    database_text_encoding: int
    user_version: int
    is_incremental_vacuum_mode: int
    application_id: int
    version_valid_for_number: int
    sqlite_version_number: int


@dataclass(frozen=True)
class BTreeHeader:
    # 2,5,10,13
    type_flag: int
    first_freeblock: int
    num_of_cells: int
    cell_content: int
    num_of_fragmented_free_bytes: int
    right_most_pointer: Optional[int] = None


def parse_file_header(raw_header: bytes) -> FileHeader:
    parsed = unpack(">16shbbbbbbIIIIIIIIIIII20xII", raw_header)
    return FileHeader(*parsed)


def parse_btree_header(raw_header: bytes) -> BTreeHeader:
    type_flag = unpack(">b", raw_header[:1])[0]
    if type_flag in (0x02, 0x05):
        parsed = unpack(">bhhhbI", raw_header[:12])
    elif type_flag in (0x0A, 0x0D):
        parsed = unpack(">bhhhb", raw_header[:8])
    return BTreeHeader(*parsed)


database_file_path = sys.argv[1]
command = sys.argv[2]

if command == ".dbinfo":
    with open(database_file_path, "rb") as database_file:
        # You can use print statements as follows for debugging, they'll be visible when running tests.
        print("Logs from your program will appear here!")

        file_header = parse_file_header(database_file.read(HEADER_SIZE))
        print(f"database page size: {file_header.page_size}")

        b_tree_header = parse_btree_header(database_file.read(12))
        print(f"number of tables: {b_tree_header.num_of_cells}")

else:
    print(f"Invalid command: {command}")
