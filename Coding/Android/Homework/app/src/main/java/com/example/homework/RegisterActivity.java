package com.example.homework;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import com.mob.MobSDK;

public class RegisterActivity extends AppCompatActivity {
    private Button register;
    private Button get_verification_code;
    @Override
    public void onBackPressed(){
        Intent intent = new Intent();
        setResult(RESULT_OK,intent);
        finish();
    }
    @Override
    public void onCreate(Bundle savedInstanceState){

        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);
        Toolbar toolbar=(Toolbar)findViewById(R.id.toolbar_register);
        setSupportActionBar(toolbar);
        toolbar.setTitle("注册"); //标题
        MobSDK.init(this);
        register=(Button)findViewById(R.id.press_register);
        get_verification_code=(Button)findViewById(R.id.get_verification_code);
        get_verification_code.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

            }
        });
        register.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                //Toast.makeText(RegisterActivity.this,"注册成功",Toast.LENGTH_SHORT);
                Intent intent = new Intent(RegisterActivity.this,LoginActivity.class);
                startActivity(intent);
                finish();
            }
        });
        //  toolbar.setSubtitle("");  副标题
        setSupportActionBar(toolbar);
    }
}
