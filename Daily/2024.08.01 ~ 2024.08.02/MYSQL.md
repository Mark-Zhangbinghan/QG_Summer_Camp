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



##### 数据的导入导出：

```mysql
-- 导出数据
	-- 如果表的名称player省略，则会导出数据库game中的所有数据
	-- 将数据导出到game.sql文件中
mysqldump -u root -p game player > game.sql  

-- 打开文件
ls -ltr
cat game.sql

-- 导入数据
mysql -u root -p game < game.sql
```



##### 常见语句：

|  名称   |          用法          |                             代码                             |
| :-----: | :--------------------: | :----------------------------------------------------------: |
|  WHERE  | 用来提取满足标准的记录 |      SELECT * FROM player where  level>1 AND level<5；       |
|   IN    |       指定多个值       |        SELECT * FROM player where level IN (1,3,5)；         |
| BETWEEN |        指定范围        |     SELECT * FROM player where level BETWEEN 1 AND 10；      |
|   NOT   |          取否          |      SELECT * FROM player where level NOT IN (1,3,5)；       |
|  LIKE   |        模糊查询        | SELECT * FROM player where name LIKE '%王%'；    SELECT * FROM player where name LIKE '王_'； |
| REGEXP  |       正则表达式       | SELECT * FROM player where name REGEXP '^王.$'；SELECT * FROM player where name REGEXP '王\|张'； |

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202408010942721.png)

|    名称    |    用法    |                             代码                             |
| :--------: | :--------: | :----------------------------------------------------------: |
|    NULL    |    空值    |     email is null or email = ''   &&   email is not null     |
|  ORDER BY  |    排列    |       SELECT * FROM player ORDER BY level ASC / DESC;        |
|  GROUP BY  |    分组    |      SELECT level, count(*) FROM player GROUP BY level;      |
|   HAVING   |    筛选    | SELECT level, count(*) FROM player GROUP BY level HAVING count(level) > 4; |
|   SUBSTR   | 截取字符串 | SELECT SUBSTR(name, 1, 1), COUNT(SUBSTR(name, 1, 1)) FROM player GROUP BY SUBSTR(name, 1, 1) HAVING COUNT(SUBSTR(name, 1, 1)) >= 5 ORDER BY COUNT(SUBSTR(name, 1, 1)) DESC; |
|   EXISTS   |    存在    |    SELECT EXISTS((select * from player where level > 10);    |
|  DISTINCT  |    去重    |               SELECT DISTINCT sex FROM player;               |
|   UNION    |    并集    | SELECT * FROM player where level BETWEEN 1 AND 3 UNION SELECT * FROM player where exp BETWEEN 1 AND 3； |
| INSTERSECT |    交集    | SELECT * FROM player where level BETWEEN 1 AND 3 INTERSECT SELECT * FROM player where exp BETWEEN 1 AND 3； |
|   EXCEPT   |    差集    | SELECT * FROM player where level BETWEEN 1 AND 3 EXCEPT SELECT * FROM player where exp BETWEEN 1 AND 3； |

常见的聚合函数：

![](https://cdn.jsdelivr.net/gh/Mark-Zhangbinghan/QG_Summer_Camp@main/picture/202408011001884.png)

```mysql
select count(*) from player;
select avg(level) from player;
```



##### 子查询：

```mysql
-- 将select语句进行嵌套
select * from player where level > (select avg(level) from player);

-- as也可以对列名进行定义
select level, ROUND(select avg(level) from plyer) as average, 
from player;

-- 用查询到的数据建立新表
	-- 还未创建
create table new_player select * from player wherer level < 5;
	-- 已经创建
insert into new_player select * from player where level between 6 and 10;
```



##### 表关联：

```mysql
/*
INNER JOIN 内连接，只返回两个表中都有的数据
LEFT JOIN 左连接，返回左表中所有的数据与右表中匹配的数据，右表中没有的数据用NULL填充
RIGHT JOIN 右连接，返回右表中所有的数据与左表中匹配的数据，左表中没有的数据用NULL填充
*/

select * from player inner join equip on player.id = equip.player_id;
```



##### 索引：

select在查找数据的时候由于是对表中数据从头到尾进行遍历，所以在数据量过大时查询效率过低
这时我们引入索引，来方便我们的查找

```mysql
create [unique(唯一索引), fulltext(全文索引), special(空间索引)] index index_name on table_name (index_col_name, ...)

-- 查看索引
show index from table_name

-- 删除索引
drop index index_name on table_name
```

