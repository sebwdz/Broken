#!/usr/bin/python3.5

from os import listdir
from os.path import isfile, join

import psycopg2

import begin


@begin.start
def main(source, image_type, details):
    files = [f for f in listdir(source) if isfile(join(source, f)) and ".json" not in f]
    values = [{"filename": file, "image_type": image_type, "details": details} for file in files]
    query = "INSERT INTO images (filename, image_type, details) VALUES (%(filename)s, %(image_type)s, %(details)s)"
    query += " ON CONFLICT (filename) DO UPDATE SET image_type = excluded.image_type, details = excluded.details"
    conn = psycopg2.connect("dbname='watson' user='sebastien' host='localhost' password='password'")
    cur = conn.cursor()
    cur.executemany(query, values)
    conn.commit()
    conn.close()
