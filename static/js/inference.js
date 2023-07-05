// 获取图片的唯一编码
let img_key;
// 点击显示图片并预览
$(document).on("click","#upload_img", function () {
    //点击图片的同时上传文件
    $('.file-upload').click();
    $(document).on("change", ".file-upload",function () {
        var fileObj = $(".file-upload")[0];
        var img = document.getElementById('upload_img');
        var reader = new FileReader();
        reader.onload = function (e) {
            if (reader.readyState === 2) {
                img.src = e.target.result;
                console.log(img.src);
            }
        }
        reader.readAsDataURL(fileObj.files[0]);

    });
});
// 点击按钮上传图片
// $(document).on("click", "#btn", function () {
//     var imgFile = new FileReader()
//     // 获取图片元素
//     var image = $("#upload_img")[0].src;
//     // 创建一个FormData对象, 并将图片添加到其中
//     var formData = new FormData();
//     formData.append('image', image)
//
//     //发起请求
//     //获取当前模型的路径
//     var currentURL = window.location.href;
//     var pathParts = currentURL.split('/');
//     var model_path = pathParts[3];
//     $.ajax = function (model_path, formData) {
//         $.ajax
//         ({
//             url: '/' + model_path.toString() + '/data_handel',
//             type: 'POST',
//             data: formData,
//             async: false,
//             success: function (res) {
//                 alert('图片上传成功');
//             }
//         });
//     }
// });

$(document).ready(function() {
  // 绑定按钮点击事件: 上传
  $('#btn_upload').click(function() {
    // 获取图片元素
    var image = $("#upload_img")[0].src;
    // 创建一个FormData对象, 并将图片添加到其中
    var formData = new FormData();
    formData.append('image', image)

    var currentURL = window.location.href;
    var pathParts = currentURL.split('/');
    var model_path = pathParts[3];
    // var img = document.getElementById('download_img');

    // 发起AJAX请求
    $.ajax({
        url: '/'+model_path+'/data_handle', // Flask的/upload接口的URL
        type: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        success: function(response) {
            // 上传成功后的操作
            alert(response['apply']);
            img_key =response['key'];
            // alert('图片上传成功！');
        //console.log(response); // 可根据需要处理服务器返回的响应数据
      }
    });

  });// 绑定按钮点击事件: 下载推理后的图片
  $('#btn_download').click(function() {
    // 获取图片元素
    // var image = $('#download_img')[0].src;
    var img = document.getElementById('download_img');

    var currentURL = window.location.href;
    var pathParts = currentURL.split('/');
    var model_path = pathParts[3];
    // 发起AJAX请求
    var input = {
      'key': img_key,
    };
    $.ajax({
        url: '/'+model_path+'/data_handle/res', // Flask的/upload接口的URL
        type: 'POST',
        data: JSON.stringify(input),
        processData: false,
        contentType: 'application/json',
        // contentType: false,
        success: function(response) {
            // 上传成功后的操作
            img.src = "data:;base64,"+response
            // image = "data:;base64,"+response
            alert('图片下载成功！');
        //console.log(response); // 可根据需要处理服务器返回的响应数据
      }
    });

  });

});