custom_css = """
body {
    /* 加载背景图 */
    background-image: url(https://i.ibb.co/F4hKFrZ/dark-miku.webp);
    /* 背景图垂直、水平均居中 */
    background-position: center center;
    /* 背景图不平铺 */
    background-repeat: no-repeat;
    /* 当内容高度大于图片高度时，背景图像的位置相对于viewport固定 */
    background-attachment: fixed;
    /* 让背景图基于容器大小伸缩 */
    background-size: cover;
    /* 设置背景颜色，背景图加载过程中会显示背景色 */
    background-color: #464646;
}

.markdown-text {
    font-size: 16px !important;
}

#models-to-add-text {
    font-size: 18px !important;
}

#citation-button span {
    font-size: 16px !important;
}

#citation-button textarea {
    font-size: 16px !important;
}

#citation-button > label > button {
    margin: 6px;
    transform: scale(1.3);
}

#leaderboard-table {
    margin-top: 15px
}

#leaderboard-table-lite {
    margin-top: 15px
}

#search-bar-table-box > div:first-child {
    background: none;
    border: none;
}
 
#search-bar {
    padding: 0px;
}

/* Limit the width of the first AutoEvalColumn so that names don't expand too much */
table td:first-child,
table th:first-child {
    max-width: 400px;
    overflow: auto;
    white-space: nowrap;
}

.tab-buttons button {
    font-size: 20px;
}

#scale-logo {
    border-style: none !important;
    box-shadow: none;
    display: block;
    margin-left: auto;
    margin-right: auto;
    max-width: 600px;
}

#scale-logo .download {
    display: none;
}
#filter_type{
    border: 0;
    padding-left: 0;
    padding-top: 0;
}
#filter_type label {
    display: flex;
}
#filter_type label > span{
    margin-top: var(--spacing-lg);
    margin-right: 0.5em;
}
#filter_type label > .wrap{
    width: 103px;
}
#filter_type label > .wrap .wrap-inner{  
    padding: 2px;
}
#filter_type label > .wrap .wrap-inner input{
    width: 1px
}
#filter-columns-type{
    border:0;
    padding:0.5;
}
#filter-columns-size{
    border:0;
    padding:0.5;
}
#box-filter > .form{
    border: 0
}
"""

get_window_url_params = """
    function(url_params) {
        const params = new URLSearchParams(window.location.search);
        url_params = Object.fromEntries(params);
        return url_params;
    }
    """
