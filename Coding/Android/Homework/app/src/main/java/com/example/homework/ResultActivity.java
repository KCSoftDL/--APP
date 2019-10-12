package com.example.homework;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.os.Bundle;
import android.widget.ImageView;
import java.io.FileInputStream;
import java.io.FileNotFoundException;

public class ResultActivity extends Activity {
    @Override
    protected void onCreate( Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.photo_result);
        String path=getIntent().getStringExtra("picpath");//通过值"picpath"得到照片路径
        ImageView imageview=findViewById(R.id.pic);
        try{FileInputStream fis=new FileInputStream(path);//通过path把照片读到文件输入流中
            Bitmap bitmap=BitmapFactory.decodeStream(fis);//将输入流解码为bitmap
            Matrix matrix=new Matrix();//新建一个矩阵对象
            matrix.setRotate(270);//矩阵旋转操作让照片可以正对着你。但是还存在一个左右对称的问题
//新建位图，第2个参数至第5个参数表示位图的大小，matrix中是旋转后的位图信息，并使bitmap变量指向新的位图对象
            bitmap=Bitmap.createBitmap(bitmap,0,0,bitmap.getWidth(),bitmap.getHeight(),matrix,true);
            //将位图展示在imageview上
            imageview.setImageBitmap(bitmap);}
        catch (FileNotFoundException e){e.printStackTrace();}
        //Bitmap bitmap=BitmapFactory.decodeFile(path);
    }
}
