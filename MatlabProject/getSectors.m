function [ dict ] = getSectors()

dict = containers.Map;

dict('SPX Index')    = {'S5COND Index', 'S5CONS Index', 'S5ENRS Index', ...
    'S5FINL Index', 'S5HLTH Index', 'S5INDU Index', 'S5INFT Index', ...
    'S5MATR Index', 'S5TELS Index'};

dict('S5COND Index') = {'S5AUCO Index', 'S5CODU Index', 'S5HOTR Index', ...
    'S5MEDA Index', 'S5RETL Index'};

dict('S5CONS Index') = {'S5FDSR Index', 'S5FDBT Index','S5HOUS Index'};

dict('S5ENRS Index') = {'S5ENRSX Index'};

dict('S5FINL Index') = {'S5BANKX Index','S5DIVF Index','S5INSU Index', 'S5REAL Index'};

dict('S5HLTH Index') = {'S5HCES Index','S5PHRM Index'};

dict('S5INDU Index') = {'S5CPGS Index', 'S5COMS Index', 'S5TRAN Index'};

dict('S5INFT Index') = {'S5SFTW Index','S5TECH Index','S5SSEQ Index'};

dict('S5MATR Index') = {'S5MATRX Index'};

dict('S5TELS Index') = {'S5TELSX Index'};

dict('S5UTIL Index') = {'S5UTILX Index'};

end