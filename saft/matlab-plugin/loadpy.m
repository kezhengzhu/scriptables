pyversion('/anaconda3/bin/python3.7')

if count(py.sys.path,'') == 0
    insert(py.sys.path,int32(0),'');
end

if count(py.sys.path,'./pylib') == 0
    insert(py.sys.path,int32(0),'./pylib');
end