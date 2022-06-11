# 导入flask对象
import os
import uuid

from flask import Flask, render_template, request, send_from_directory

from werkzeug.utils import secure_filename

from com.get_data import get_traj_table, get_traj_chart
from com.utils.utils import create_dir
from com.model_test import muti_seq2seq_match

# 使用flask对象创建一个app对象
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/upload/'
app.config['DOWNLOAD_FOLDER'] = 'static/download/'
app.config['DOWN_RAW_MATCH'] = ''

# 地图上显示的原始轨迹和匹配轨迹
raw_t, match_t, raw_map, match_map = get_traj_table()

# print(raw_t)

app.raw_file, app.match_file = '', ''

app.table_point = {}
app.table_point['raw'] = raw_t
app.table_point['match'] = match_t

app.map_point = {}
app.map_point['raw'] = raw_map
app.map_point['match'] = match_map

app.chart_dict = {}
app.chart_dict = get_traj_chart('com/data/model/model_data/test_data/test_raw_trajectory_1.txt')

# 路由
@app.route("/", methods=['GET', 'POST']) # 访问路径
def root():
    """
    主页
    :return: Index.html
    """
    if (request.method == 'POST'):
        f = request.files['file']
        fname = secure_filename(f.filename)

        uid = str(uuid.uuid4())

        # 生成一个uuid作为文件夹，创建文件夹
        fileDir = app.config['UPLOAD_FOLDER'] + uid + "/"
        create_dir(fileDir)

        # os.path.join拼接地址，上传地址
        f.save(os.path.join(fileDir, fname))

        # 上传的数据统计
        app.chart_dict = get_traj_chart(fileDir + fname)

        # 进行匹配与处理, 返回分割后的原始轨迹文件以及匹配轨迹
        app.raw_file, app.match_file = muti_seq2seq_match(test_trajs_dir=fileDir, save_dir=app.config['DOWNLOAD_FOLDER']+uid+'/')

        # 下载轨迹路径
        app.config['DOWN_RAW_MATCH'] = app.config['DOWNLOAD_FOLDER']+uid+'/'

        # 统计表格和地图显示轨迹
        raw_t, match_t, raw_map, match_map = get_traj_table(n=1, traj_dir=app.config['DOWN_RAW_MATCH'])
        app.table_point = {}
        app.table_point['raw'] = raw_t
        app.table_point['match'] = match_t

        app.map_point = {}
        app.map_point['raw'] = raw_map
        app.map_point['match'] = match_map

        app.match_file = 'match.txt'

    return render_template('index.html', match_file = app.match_file)

@app.route("/<filename>", methods=['POST','GET'])
def dowmload(filename):
    if request.method == "GET":
        path = os.path.isfile(os.path.join(app.config['DOWN_RAW_MATCH'], filename))
        if path:
            return send_from_directory(app.config['DOWN_RAW_MATCH'], filename, as_attachment=True)

@app.route("/table")
def table():
    return render_template('table.html', traj = app.table_point)

@app.route("/maps")
def maps():
    return render_template('maps.html', point= app.map_point)

@app.route("/charts")
def charts():
    return render_template('charts.html', static_dict = app.chart_dict)

@app.route("/about")
def about():
    return render_template('about.html')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port='5000')

