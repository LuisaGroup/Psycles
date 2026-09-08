"""Decode the original Cycles observer's length-framed resource registries.

This reads metadata, not shader values or a host shader implementation.
Keep both registries with the SVM word dump from the same original session.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


class Reader:
    def __init__(self, data: bytes, magic: bytes):
        self.data = data
        self.offset = 0
        if self.take(8) != magic or self.u32() != 1:
            raise ValueError('unsupported original binding dump')

    def take(self, count: int) -> bytes:
        if count < 0 or count > len(self.data) - self.offset:
            raise ValueError('truncated original binding dump')
        start = self.offset
        self.offset += count
        return self.data[start:self.offset]

    def u32(self) -> int:
        return struct.unpack('<I', self.take(4))[0]

    def i32(self) -> int:
        return struct.unpack('<i', self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack('<Q', self.take(8))[0]

    def f32(self) -> float:
        value = struct.unpack('<f', self.take(4))[0]
        if not math.isfinite(value):
            raise ValueError('non-finite image frame')
        return value

    def text(self) -> str:
        return self.take(self.u32()).decode('utf-8', errors='strict')

    def present(self) -> bool:
        value = self.u32()
        if value not in (0, 1):
            raise ValueError('invalid presence flag')
        return bool(value)

    def end(self):
        if self.offset != len(self.data):
            raise ValueError('trailing original binding data')


def image_bindings(data: bytes) -> dict:
    reader = Reader(data, b'PSYIMG52')
    images = []
    for index in range(reader.u32()):
        if reader.u32() != index:
            raise ValueError('non-dense image slot sequence')
        if not reader.present():
            images.append(None)
            continue
        image = {'id': reader.i32(), 'state_flags': reader.u32(),
                 'interpolation': reader.u32(), 'extension': reader.u32(),
                 'alpha_type': reader.u32(), 'animated': reader.present(),
                 'frame': reader.f32(), 'miplevel_offset': reader.i32(),
                 'width': reader.u64(), 'height': reader.u64(),
                 'channels': reader.u32(), 'data_type': reader.u32(),
                 'metadata_flags': reader.u32(), 'tile_size': reader.u32(),
                 'loader_name': reader.text(), 'requested_colorspace': reader.text(),
                 'final_colorspace': reader.text(), 'tile_number': reader.i32()}
        if image['id'] != index or image['state_flags'] & ~7 or image['metadata_flags'] & ~63:
            raise ValueError('inconsistent native image slot or flags')
        images.append(image)
    udims = []
    for index in range(reader.u32()):
        if reader.u32() != index:
            raise ValueError('non-dense UDIM slot sequence')
        if not reader.present():
            udims.append(None)
            continue
        udim = {'id': reader.i32(), 'tiles': []}
        for _ in range(reader.u32()):
            tile, image_id = reader.i32(), reader.i32()
            if image_id < 0 or image_id >= len(images) or images[image_id] is None:
                raise ValueError('invalid UDIM image reference')
            udim['tiles'].append({'tile': tile, 'image_id': image_id})
        if udim['id'] >= 0 or len({t['tile'] for t in udim['tiles']}) != len(udim['tiles']):
            raise ValueError('invalid UDIM identity or duplicate tiles')
        udims.append(udim)
    reader.end()
    if len({u['id'] for u in udims if u is not None}) != sum(u is not None for u in udims):
        raise ValueError('duplicate UDIM identities')
    return {'images': images, 'udims': udims}


def attribute_bindings(data: bytes) -> list[dict]:
    reader = Reader(data, b'PSYATT52')
    attributes = [{'id': reader.u64(), 'name': reader.text()} for _ in range(reader.u32())]
    reader.end()
    if len({a['id'] for a in attributes}) != len(attributes) or len({a['name'] for a in attributes}) != len(attributes):
        raise ValueError('duplicate attribute identity')
    return sorted(attributes, key=lambda a: a['id'])


def psycles_bindings(data: bytes) -> dict:
    reader = Reader(data, b'PSYPBD52')
    attributes = [{'id': reader.u64(), 'name': reader.text()} for _ in range(reader.u32())]
    if len({a['id'] for a in attributes}) != len(attributes) or len({a['name'] for a in attributes}) != len(attributes):
        raise ValueError('duplicate actual attribute identity')
    images = []
    for index in range(reader.u32()):
        image = {'id': reader.u32(), 'resource_id': reader.u64(),
                 'interpolation': reader.u32(), 'extension': reader.u32()}
        if image['id'] != index or image['interpolation'] > 3 or image['extension'] > 3:
            raise ValueError('invalid actual image binding')
        if reader.present():
            image['nishita'] = {'multiple_scattering': reader.present(),
                                'parameter_bits': [reader.u32() for _ in range(5)]}
        else:
            image.update(name=reader.text(), color_space=reader.u32(),
                         alpha_type=reader.u32(), width=reader.u32(),
                         height=reader.u32(), load_failed=reader.present())
            if image['color_space'] > 2 or image['alpha_type'] > 3:
                raise ValueError('invalid actual image metadata')
        images.append(image)
    reader.end()
    return {'attributes': sorted(attributes, key=lambda a: a['id']), 'images': images}


def source(path: Path) -> dict:
    data = path.read_bytes()
    return {'path': str(path), 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('images', type=Path)
    parser.add_argument('attributes', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = {'schema': 'psycles.original-resource-bindings.v1',
              'scope': 'observed native registries; loader names alone do not prove global identity',
              'images': image_bindings(args.images.read_bytes()),
              'attributes': attribute_bindings(args.attributes.read_bytes()),
              'sources': [source(args.images), source(args.attributes)]}
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
