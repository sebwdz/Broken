#!/usr/bin/python3.5

import json
import psycopg2

import begin


@begin.start
def main(source):
    bbox = json.load(open(source))
    conn = psycopg2.connect("dbname='watson' user='sebastien' host='localhost' password='password'")
    cur = conn.cursor()
    cur.execute('SELECT id, filename FROM images')
    images = {x[1]: x[0] for x in cur.fetchall()}
    query = "INSERT INTO boundingbox (fk_image_id, settings) VALUES (%(fk_image_id)s, %(settings)s)"
    batch = []
    for img in bbox:
        img_id = images[img['filename']]
        for box in img['annotations']:
            batch.append({"fk_image_id": img_id, "settings": json.dumps(box)})
    cur.executemany(query, batch)
    conn.commit()
    cur = conn.cursor()
    cur.execute("UPDATE boundingbox SET fk_parent_id = t.id FROM (SELECT id, fk_image_id FROM boundingbox " +
                "WHERE boundingbox.settings->>'class' = 'rect') as t WHERE boundingbox.settings->>'class' = 'Face' "+
                "AND boundingbox.fk_image_id = t.fk_image_id")
    conn.commit()

