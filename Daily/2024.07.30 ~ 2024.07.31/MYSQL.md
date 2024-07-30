## MYSQL

#### 安装：

直接官网查找安装包即可



#### MySQL-Shell：(所有的指令都是由'\'开头的)

在官网下载安装包即可（也可以直接在VS Code中找它的插件，其内部类似与jupyter notebook）

```mysql
-- 进入mysql-shell环境
~/mysql> mysqlsh

-- 查看帮助信息
MySQL JS> \help

-- 连接到服务器
MySQL JS> connect root@localhost -- 连接到本地

-- 连接到game数据库
MySQL localhost:33036+ ssl JS> \use game

-- 切换语言
MySQL localhost:33036+ ssl game JS> \py
MySQL localhost:33036+ ssl game Py> print("hello")
MySQL localhost:33036+ ssl game Py> \sql
-- 执行SQL语句
MySQL localhst:33036+ ssl game SQL> show tables; -- 列出所有在数据库中的表
```



#### SQL基础:

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202407302231993.png)



##### 创建数据库：

```mysql
-- 进入mysql环境
>>mysql -u root -p
>>"这里输入密码"

-- 清空屏幕
mysql> \! clear

-- 先查看已有的数据库
mysql> show databases;

-- 创建数据库
mysql> create database game；

-- 删除数据库
mysql> drop database game;
```



##### 创建表：

```mysql
-- 进入数据库
mysql> use game;

-- 建立表
mysql game> create table player (
			id INT,
			name VARCHAR(100),
    		level INT,
			exp INT,
    		gold DECIMAL(10,2)
)

-- 描述表的结构
mysql game> DESC player;

-- 修改表
	-- 修改字段类型
mysql game> ALTER table player MODIFY COLUMN name VARCHAR(200);
	-- 修改字段名称
mysql game> ALTER tabel player RENAME COLUMN name to nick_name;
	-- 添加字段
mysql game> ALTER tabel player ADD COLUMN last_login DATETIME;
	-- 删除字段
mysql game> ALTER tabel player DROP COLUMN last_login;
	-- 设置字段的默认值
mysql game> ALTER TABLE player MODIFY LEVEL INT DEFAULT 1;

-- 删除表
mysql game> DROP TABLE player；
```



##### 数据的增删改查：

```mysql
-- 先创建表
mysql game> create table player (
			id INT,
			name VARCHAR(100),
    		level INT,
			exp INT,
    		gold DECIMAL(10,2)
)

-- 增加数据
mysql game> INSERT INTO player (id, name, level, exp, gold) VALUES (1, '张三'， 1， 1， 1)；

-- 查询数据
mysql game> select * from player -- * 表示查找表中的所有数据，player 为表名

-- 修改数据
	-- 仅仅修改张三
mysql game> UPDATE player set level = 1 where name = '张三' ；
	-- 一次性全部修改
mysql game> UPDATE player set level = 1，gold = 0； 

-- 删除数据
mysql game> DELETE FROM player where gold=0; -- 删除所有金钱为0的玩家
```

