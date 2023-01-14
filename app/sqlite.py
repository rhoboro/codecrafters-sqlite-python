import itertools
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from struct import unpack
from typing import Iterable, List, Optional, Tuple, Union

HEADER_SIZE = 100
ValueType = Union[str, int, bytes]


class BTreeType(IntEnum):
    TableLeafCell = 0x0D
    TableInteriorCell = 0x05
    IndexLeafCell = 0x0A
    IndexInteriorCell = 0x02


@dataclass(frozen=True)
class BTreeHeader:
    type_flag: BTreeType
    first_free_block: int
    num_of_cells: int
    cell_content: int
    num_of_fragmented_free_bytes: int
    right_most_pointer: Optional[int] = None

    @property
    def size(self) -> int:
        return 8 if self.right_most_pointer is None else 12


class SerialTypeCode(Enum):
    null = auto()
    int_8bit = auto()
    int_16bit = auto()
    int_24bit = auto()
    int_32bit = auto()
    int_48bit = auto()
    int_64bit = auto()
    float_64bit = auto()
    int_0 = auto()
    int_1 = auto()
    internal = auto()
    blob = auto()
    string = auto()

    @classmethod
    def get_code_and_size(cls, num: int) -> Tuple["SerialTypeCode", int]:
        code_and_size = SerialTypeCodeMap.get(num)
        if code_and_size is not None:
            return code_and_size[0], code_and_size[1]

        if num == 10 or num == 11:
            # TODO:
            raise NotImplementedError()

        if num % 2 == 0:
            return SerialTypeCode.blob, (num - 12) // 2
        else:
            return SerialTypeCode.string, (num - 13) // 2


SerialTypeCodeMap = {
    0: (SerialTypeCode.null, 0),
    1: (SerialTypeCode.int_8bit, 1),
    2: (SerialTypeCode.int_16bit, 2),
    3: (SerialTypeCode.int_24bit, 3),
    4: (SerialTypeCode.int_32bit, 4),
    5: (SerialTypeCode.int_48bit, 6),
    6: (SerialTypeCode.int_64bit, 8),
    7: (SerialTypeCode.float_64bit, 8),
    8: (SerialTypeCode.int_0, 0),
    9: (SerialTypeCode.int_1, 0),
}


class BTree:
    def __init__(self, page: bytes, page_num: int, reserved_size: int = 0) -> None:
        self.page = page
        self.page_num = page_num
        self.reserved_size = reserved_size
        self.header = self.parse_header(self.page, self.page_num)
        self.cell_pointers = list(
            self.parse_cell_pointers(self.page, self.page_num, self.header)
        )
        self.rows = [
            self.parse_cell(self.page, self.page_num, self.header, cell_position)
            for cell_position in self.cell_pointers
        ]

    def parse_header(self, page: bytes, page_num: int) -> BTreeHeader:
        offset = HEADER_SIZE if page_num == 1 else 0
        type_flag = unpack(">b", page[offset : offset + 1])[0]
        if type_flag in (BTreeType.IndexInteriorCell, BTreeType.TableInteriorCell):
            header_size = 12
            parsed = unpack(">bhhhbI", page[offset : offset + header_size])
        elif type_flag in (BTreeType.IndexLeafCell, BTreeType.TableLeafCell):
            header_size = 8
            parsed = unpack(">bhhhb", page[offset : offset + header_size])
        else:
            raise SQLiteException("invalid format")

        return BTreeHeader(*parsed)

    def parse_cell_pointers(
        self, page: bytes, page_num: int, header: BTreeHeader
    ) -> Iterable[int]:
        offset = HEADER_SIZE if page_num == 1 else 0
        cell_pointers = page[
            offset + header.size : (offset + header.size) + (2 * header.num_of_cells)
        ]
        for cell_pointer in get_chunk(cell_pointers, 2):
            high, low = list(cell_pointer)
            cell_position = high << 8 | low
            yield cell_position

    def parse_cell(
        self, page: bytes, page_num: int, header: BTreeHeader, cell_position: int
    ):
        if header.type_flag == BTreeType.TableLeafCell:
            return self._parse_table_leaf_cell(page, page_num, cell_position)

        raise NotImplementedError()

    def _parse_table_leaf_cell(self, page: bytes, page_num: int, cell_position: int):
        relative_position = cell_position - ((page_num - 1) * len(page))
        bytes_of_payload, consumed_1 = get_a_varint(page[relative_position:])
        rowid, consumed_2 = get_a_varint(page[relative_position + consumed_1 :])
        row = self._read_payload(
            page[
                relative_position
                + consumed_1
                + consumed_2 : relative_position
                + consumed_1
                + consumed_2
                + bytes_of_payload
            ]
        )
        return row

    def _read_payload(
        self, payload: bytes
    ) -> List[Tuple[SerialTypeCode, int, ValueType]]:
        bytes_of_header, consumed = get_a_varint(payload)
        columns = []
        while consumed < bytes_of_header:
            serial_type, consumed_ = get_a_varint(payload[consumed:])
            columns.append(SerialTypeCode.get_code_and_size(serial_type))
            consumed += consumed_

        cols = []
        column_pos = consumed
        for col_type, size in columns:
            cols.append((col_type, size, payload[column_pos : column_pos + size]))
            column_pos += size

        return cols


class SQLiteException(Exception):
    pass


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


class Database:
    def __init__(self, database_file: str) -> None:
        self.database_file = database_file
        self.header = self._parse_header()
        # CREATE TABLE sqlite_schema(
        #   type text,
        #   name text,
        #   tbl_name text,
        #   rootpage integer,
        #   sql text
        # );
        self.sqlite_schema = BTree(self.read_pages(0, 1), 1)

    @property
    def page_size(self) -> int:
        return self.header.page_size

    @property
    def num_of_tables(self) -> int:
        return len(self.sqlite_schema.cell_pointers)

    @property
    def table_names(self) -> List[str]:
        tables = []
        tbl_name_index = 2
        for row in self.sqlite_schema.rows:
            table_name = (row[tbl_name_index][2]).decode()
            if not table_name.startswith("sqlite_"):
                tables.append(table_name)

        return tables

    def read_pages(self, start: int, num: int) -> bytes:
        with open(self.database_file, "rb") as f:
            f.seek(self.page_size * start)
            return f.read(self.page_size * num)

    def _parse_header(self) -> FileHeader:
        with open(self.database_file, "rb") as f:
            parsed = unpack(">16sHbbbbbbIIIIIIIIIIII20xII", f.read(HEADER_SIZE))
        return FileHeader(*parsed)


def get_chunk(iterable, n=5):
    for i, item in itertools.groupby(enumerate(iterable), lambda x: x[0] // n):
        yield (x[1] for x in item)


def get_a_varint(byte_list: bytes) -> Tuple[int, int]:
    """varint

    :return: (result, consumed)
    """
    consumed = 1
    value = 0b00000000
    for byte in byte_list:
        current = byte & 0b01111111
        value = value << 7 | current
        has_next = byte >> 7
        if has_next and consumed < 9:
            consumed += 1
        else:
            return value, consumed
